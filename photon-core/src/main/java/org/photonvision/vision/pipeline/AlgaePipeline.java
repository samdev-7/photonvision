package org.photonvision.vision.pipeline;

import java.util.List;
import java.util.stream.Collectors;

import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVShape;
import org.photonvision.vision.opencv.DualOffsetValues;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.Collect2dTargetsPipe;
import org.photonvision.vision.pipe.impl.Draw2dCrosshairPipe;
import org.photonvision.vision.pipe.impl.Draw2dTargetsPipe;
import org.photonvision.vision.pipe.impl.Draw3dTargetsPipe;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe.AlgaePose;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.PotentialTarget;
import org.photonvision.vision.target.TrackedTarget;

import edu.wpi.first.math.geometry.Transform3d;

public class AlgaePipeline extends CVPipeline<CVPipelineResult, AlgaePipelineSettings> {
    private final AlgaeDetectionPipe algaeDetectionPipe = new AlgaeDetectionPipe();
    private final Collect2dTargetsPipe collect2dTargetsPipe = new Collect2dTargetsPipe();
    private final Draw2dCrosshairPipe draw2dCrosshairPipe = new Draw2dCrosshairPipe();
    private final Draw2dTargetsPipe draw2DTargetsPipe = new Draw2dTargetsPipe();
    private final Draw3dTargetsPipe draw3dTargetsPipe = new Draw3dTargetsPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.NONE;

    public AlgaePipeline() {
        super(PROCESSING_TYPE);
        settings = new AlgaePipelineSettings();
    }

    public AlgaePipeline(AlgaePipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {
        DualOffsetValues dualOffsetValues =
                new DualOffsetValues(
                        settings.offsetDualPointA,
                        settings.offsetDualPointAArea,
                        settings.offsetDualPointB,
                        settings.offsetDualPointBArea);
    
        AlgaeDetectionPipe.AlgaeDetectionParams algaeDetectionParams = new AlgaeDetectionPipe.AlgaeDetectionParams();
        algaeDetectionPipe.setParams(algaeDetectionParams);

        Collect2dTargetsPipe.Collect2dTargetsParams collect2dTargetsParams =
                new Collect2dTargetsPipe.Collect2dTargetsParams(
                        settings.offsetRobotOffsetMode,
                        settings.offsetSinglePoint,
                        dualOffsetValues,
                        settings.contourTargetOffsetPointEdge,
                        settings.contourTargetOrientation,
                        frameStaticProperties);
        collect2dTargetsPipe.setParams(collect2dTargetsParams);
        
        Draw2dTargetsPipe.Draw2dTargetsParams draw2DTargetsParams =
                new Draw2dTargetsPipe.Draw2dTargetsParams(
                        settings.outputShouldDraw,
                        settings.outputShowMultipleTargets,
                        settings.streamingFrameDivisor);
        draw2DTargetsParams.showShape = true;
        draw2DTargetsParams.showMaximumBox = false;
        draw2DTargetsParams.showRotatedBox = false;
        draw2DTargetsPipe.setParams(draw2DTargetsParams);

        Draw2dCrosshairPipe.Draw2dCrosshairParams draw2dCrosshairParams =
                new Draw2dCrosshairPipe.Draw2dCrosshairParams(
                        settings.outputShouldDraw,
                        settings.offsetRobotOffsetMode,
                        settings.offsetSinglePoint,
                        dualOffsetValues,
                        frameStaticProperties,
                        settings.streamingFrameDivisor,
                        settings.inputImageRotationMode);
        draw2dCrosshairPipe.setParams(draw2dCrosshairParams);

        var draw3dTargetsParams =
                new Draw3dTargetsPipe.Draw3dContoursParams(
                        settings.outputShouldDraw,
                        frameStaticProperties.cameraCalibration,
                        settings.targetModel,
                        settings.streamingFrameDivisor);
        draw3dTargetsPipe.setParams(draw3dTargetsParams);
    }

    @Override
    protected CVPipelineResult process(Frame frame, AlgaePipelineSettings settings) {
        long sumPipeNanosElapsed = 0;

        CVPipeResult<List<AlgaePose>> algaeDetectionResult = algaeDetectionPipe.run(frame.colorImage.getMat());
        sumPipeNanosElapsed += algaeDetectionResult.nanosElapsed;
        List<AlgaePose> algaePoses = algaeDetectionResult.output;
        
        long currentTimeNanos = System.nanoTime();
        List<CVShape> algaeShapes = algaePoses.stream().map(AlgaePose::getShape).collect(Collectors.toList());

        List<PotentialTarget> potentialTargets = algaeShapes.stream().map(shape -> {
            return new PotentialTarget(shape.getContour(), shape);
        }).collect(Collectors.toList());
        sumPipeNanosElapsed += System.nanoTime() - currentTimeNanos;

        CVPipeResult<List<TrackedTarget>> collect2dTargetsResult = collect2dTargetsPipe.run(potentialTargets);
        sumPipeNanosElapsed += collect2dTargetsResult.nanosElapsed;

        currentTimeNanos = System.nanoTime();
        List<TrackedTarget> targetList = collect2dTargetsResult.output;

        for (int i = 0; i < targetList.size(); i++) {
            targetList.get(i).setBestCameraToTarget3d(algaePoses.get(i).getCameraToAlgaeTransform());
            targetList.get(i).setAltCameraToTarget3d(new Transform3d());
        }

        frame.processedImage.copyFrom(frame.colorImage.getMat());

        sumPipeNanosElapsed += System.nanoTime() - currentTimeNanos;

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(frame.sequenceID, sumPipeNanosElapsed, fps, targetList, frame);
    }
}
