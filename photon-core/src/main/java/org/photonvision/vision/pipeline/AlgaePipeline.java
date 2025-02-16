package org.photonvision.vision.pipeline;

import java.util.List;

import org.opencv.core.Mat;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

public class AlgaePipeline extends CVPipeline<CVPipelineResult, AlgaePipelineSettings> {
    private final AlgaeDetectionPipe algaeDetectionPipe = new AlgaeDetectionPipe();
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
        AlgaeDetectionPipe.AlgaeDetectionParams algaeDetectionParams = new AlgaeDetectionPipe.AlgaeDetectionParams();
        algaeDetectionPipe.setParams(algaeDetectionParams);
    }

    @Override
    protected CVPipelineResult process(Frame frame, AlgaePipelineSettings settings) {
        long sumPipeNanosElapsed = 0;

        CVPipeResult<Mat> algaeDetectionResult = algaeDetectionPipe.run(frame.colorImage.getMat());
        sumPipeNanosElapsed += algaeDetectionResult.nanosElapsed;
        
        algaeDetectionResult.output.copyTo(frame.processedImage.getMat());

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;


        return new CVPipelineResult(frame.sequenceID, sumPipeNanosElapsed, fps, List.of(), frame);
    }
}
