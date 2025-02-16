package org.photonvision.vision.pipeline;

import java.util.List;

import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

public class PythonPipeline extends CVPipeline<CVPipelineResult, PythonPipelineSettings> {
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.NONE;

    public PythonPipeline() {
        super(PROCESSING_TYPE);
        settings = new PythonPipelineSettings();
    }

    public PythonPipeline(PythonPipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {

    }

    @Override
    protected CVPipelineResult process(Frame frame, PythonPipelineSettings settings) {
        long sumPipeNanosElapsed = 0;

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        frame.colorImage.getMat().copyTo(frame.processedImage.getMat());

        return new CVPipelineResult(frame.sequenceID, sumPipeNanosElapsed, fps, List.of(), frame);
    }
}
