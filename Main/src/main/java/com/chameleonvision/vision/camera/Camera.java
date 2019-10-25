package com.chameleonvision.vision.camera;

import com.chameleonvision.Main;
import com.chameleonvision.settings.Platform;
import com.chameleonvision.settings.SettingsManager;
import com.chameleonvision.vision.Pipeline;
import com.chameleonvision.web.ServerHandler;
import edu.wpi.cscore.*;
import edu.wpi.first.cameraserver.CameraServer;
import org.opencv.core.Mat;

import java.nio.channels.Pipe;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Camera {

    private static final double DEFAULT_FOV = 60.8;
    private static final StreamDivisor DEFAULT_STREAMDIVISOR = StreamDivisor.none;
    private static final int MINIMUM_FPS = 30;
    private static final int MINIMUM_WIDTH = 320;
    private static final int MINIMUM_HEIGHT = 200;
    private static final int MAX_INIT_MS = 1500;
    private static final List<VideoMode.PixelFormat> ALLOWED_PIXEL_FORMATS = Arrays.asList(VideoMode.PixelFormat.kYUYV, VideoMode.PixelFormat.kMJPEG);

    public final String name;
    public final String path;

    private String nickname;

    private final UsbCamera UsbCam;
    private final VideoMode[] availableVideoModes;

    private final CameraServer cs = CameraServer.getInstance();
    private final CvSink cvSink;
    private final Object cvSourceLock = new Object();
    private CvSource cvSource;
    private Double FOV;
    private StreamDivisor streamDivisor;
    private CameraValues camVals;
    private CamVideoMode camVideoMode;
    private int currentPipelineIndex;
    private List<Pipeline> pipelines;

    public Camera(String cameraName) {
        this(cameraName, DEFAULT_FOV);
    }

    public Camera(String cameraName, double fov) {
        this(cameraName, CameraManager.AllUsbCameraInfosByName.get(cameraName), fov);
    }

    public Camera(String cameraName, UsbCameraInfo usbCameraInfo, double fov) {
        this(cameraName, usbCameraInfo, fov, DEFAULT_STREAMDIVISOR);
    }

    public Camera(String cameraName, UsbCameraInfo usbCamInfo, double fov, StreamDivisor divisor) {
        this(cameraName, usbCamInfo, fov, new ArrayList<>(), 0, divisor);
    }

    public Camera(String cameraName, double fov, List<Pipeline> pipelines, int videoModeIndex, StreamDivisor divisor) {
        this(cameraName, CameraManager.AllUsbCameraInfosByName.get(cameraName), fov, pipelines, videoModeIndex, divisor);
    }
    public Camera(String cameraName, double fov, int videoModeIndex, StreamDivisor divisor) {
        this(cameraName, fov, new ArrayList<>(), videoModeIndex, divisor);
    }
    public Camera(String cameraName, UsbCameraInfo usbCamInfo, double fov, List<Pipeline> pipelines, int videoModeIndex, StreamDivisor divisor) {
        FOV = fov;
        name = cameraName;

        if (Platform.getCurrentPlatform().isWindows()) {
            path = usbCamInfo.path;
        } else {
            var truePath = Arrays.stream(usbCamInfo.otherPaths).filter(x -> x.contains("/dev/v4l/by-path")).findFirst();
            path = truePath.isPresent() ? truePath.toString() : null;
        }

        streamDivisor = divisor;
        UsbCam = new UsbCamera(name, path);

        this.pipelines = pipelines;

        // set up video modes according to minimums
        if (Platform.getCurrentPlatform() == Platform.WINDOWS_64 && !UsbCam.isConnected()) {
            System.out.print("Waiting on camera... ");
            long initTimeout = System.nanoTime();
            while (!UsbCam.isConnected()) {
                if (((System.nanoTime() - initTimeout) / 1e6) >= MAX_INIT_MS) {
                    break;
                }
            }
            var initTimeMs = (System.nanoTime() - initTimeout) / 1e6;
            System.out.printf("Camera initialized in %.2fms\n", initTimeMs);
        }
        var trueVideoModes = UsbCam.enumerateVideoModes();
        availableVideoModes = Arrays.stream(trueVideoModes).filter(v ->
                v.fps >= MINIMUM_FPS && v.width >= MINIMUM_WIDTH && v.height >= MINIMUM_HEIGHT && ALLOWED_PIXEL_FORMATS.contains(v.pixelFormat)).toArray(VideoMode[]::new);
        if (availableVideoModes.length == 0) {
            System.err.println("Camera not supported!");
            throw new RuntimeException(new CameraException(CameraException.CameraExceptionType.BAD_CAMERA));
        }
        if (videoModeIndex <= availableVideoModes.length - 1) {
            setCamVideoMode(videoModeIndex, false);
        } else {
            setCamVideoMode(0, false);
        }

        cvSink = cs.getVideo(UsbCam);
        cvSource = cs.putVideo(name, camVals.ImageWidth, camVals.ImageHeight);
    }

    VideoMode[] getAvailableVideoModes() {
        return availableVideoModes;
    }

    public int getStreamPort() {
        var s = (MjpegServer) cs.getServer("serve_" + name);
        return s.getPort();
    }

    public void setCamVideoMode(int videoMode, boolean updateCvSource) {
        setCamVideoMode(new CamVideoMode(availableVideoModes[videoMode]), updateCvSource);
    }

    private void setCamVideoMode(CamVideoMode newVideoMode, boolean updateCvSource) {
        var prevVideoMode = this.camVideoMode;
        this.camVideoMode = newVideoMode;
        UsbCam.setVideoMode(newVideoMode.getActualPixelFormat(), newVideoMode.width, newVideoMode.height, newVideoMode.fps);

        // update camera values
        camVals = new CameraValues(this);
        if (prevVideoMode != null && !prevVideoMode.equals(newVideoMode) && updateCvSource) { //  if resolution changed
            synchronized (cvSourceLock) {
                cvSource = cs.putVideo(name, newVideoMode.width, newVideoMode.height);
            }
            ServerHandler.sendFullSettings();
        }
    }
    public void addPipeline() {
        Pipeline p = new Pipeline();
        p.nickname = "New pipeline " + pipelines.size();
        addPipeline(p);
    }
    public void addPipeline(Pipeline p){
        this.pipelines.add(p);
    }

    public void deletePipeline(int index) {
        pipelines.remove(index);
    }
    public void deletePipeline() {
        deletePipeline(getCurrentPipelineIndex());
    }

    public Pipeline getCurrentPipeline() {
        return getPipelineByIndex(currentPipelineIndex);
    }

    public Pipeline getPipelineByIndex(int pipelineIndex) {
        return pipelines.get(pipelineIndex);
    }

    public int getCurrentPipelineIndex() {
        return currentPipelineIndex;
    }

    public void setCurrentPipelineIndex(int pipelineNumber) {
        if (pipelineNumber - 1 > pipelines.size()) return;
        currentPipelineIndex = pipelineNumber;
    }

    public StreamDivisor getStreamDivisor() {
        return streamDivisor;
    }

    public void setStreamDivisor(int divisor) {
        streamDivisor = StreamDivisor.values()[divisor];
    }

    public List<Pipeline> getPipelines() {
        return pipelines;
    }
    public List<String> getPipelinesNickname(){
        var pipelines = getPipelines();
        return pipelines.stream().map(pipeline -> pipeline.nickname).collect(Collectors.toList());
    }

    public CamVideoMode getVideoMode() {
        return camVideoMode;
    }

    public int getVideoModeIndex() {
        return IntStream.range(0, availableVideoModes.length)
                .filter(i -> camVideoMode.equals(availableVideoModes[i]))
                .findFirst()
                .orElse(-1);
    }

    public double getFOV() {
        return FOV;
    }

    public void setFOV(Number fov) {
        FOV = fov.doubleValue();
        camVals = new CameraValues(this);
    }

    public int getBrightness() {
        return getCurrentPipeline().brightness;
    }

    public void setBrightness(int brightness) {
        getCurrentPipeline().brightness = brightness;
        UsbCam.setBrightness(brightness);
    }

    public void setExposure(int exposure) {
        getCurrentPipeline().exposure = exposure;
        UsbCam.setExposureManual(exposure);
    }

    public long grabFrame(Mat image) {
        return cvSink.grabFrame(image);
    }

    public CameraValues getCamVals() {
        return camVals;
    }

    public void putFrame(Mat image) {
        synchronized (cvSourceLock) {
            cvSource.putFrame(image);
        }
    }

    public List<String> getResolutionList() {
        return Arrays.stream(availableVideoModes)
                .map(res -> String.format("%sx%s@%sFPS, %s", res.width, res.height, res.fps, res.pixelFormat.toString()))
                .collect(Collectors.toList());
    }

    public void setNickname(String newNickname) {
        nickname = newNickname;
    }

    public String getNickname() {
        return nickname == null ? name : nickname;
    }
}
