package org.photonvision.vision.pipeline;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.opencv.core.Mat;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVShape;
import org.photonvision.vision.opencv.DualOffsetValues;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.Collect2dTargetsPipe;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe.AlgaePose;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.PotentialTarget;
import org.photonvision.vision.target.TrackedTarget;

import edu.wpi.first.math.Pair;
import edu.wpi.first.math.geometry.Transform3d;

public class AlgaePipeline extends CVPipeline<CVPipelineResult, AlgaePipelineSettings> {
    private final AlgaeDetectionPipe algaeDetectionPipe = new AlgaeDetectionPipe();
    private final Collect2dTargetsPipe collect2dTargetsPipe = new Collect2dTargetsPipe();
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
    }

    @Override
    protected CVPipelineResult process(Frame frame, AlgaePipelineSettings settings) {
        long sumPipeNanosElapsed = 0;

        CVPipeResult<Pair<Mat, Optional<AlgaePose>>> algaeDetectionResult = algaeDetectionPipe.run(frame.colorImage.getMat());
        sumPipeNanosElapsed += algaeDetectionResult.nanosElapsed;

        algaeDetectionResult.output.getFirst().copyTo(frame.processedImage.getMat());

        var algaePose = algaeDetectionResult.output.getSecond();

        List<AlgaePose> algaePoses = List.of();
        if (algaePose.isPresent()) {
            algaePoses = List.of(algaePose.get());
        }
        
        List<CVShape> algaeShapes = algaePoses.stream().map(AlgaePose::getShape).collect(Collectors.toList());

        List<PotentialTarget> potentialTargets = algaeShapes.stream().map(shape -> {
            return new PotentialTarget(shape.getContour(), shape);
        }).collect(Collectors.toList());

        CVPipeResult<List<TrackedTarget>> collect2dTargetsResult = collect2dTargetsPipe.run(potentialTargets);
        sumPipeNanosElapsed += collect2dTargetsResult.nanosElapsed;

        List<TrackedTarget> targetList = collect2dTargetsResult.output;

        for (int i = 0; i < targetList.size(); i++) {
            targetList.get(i).setBestCameraToTarget3d(algaePoses.get(i).getCameraToAlgaeTransform());
            targetList.get(i).setAltCameraToTarget3d(new Transform3d());
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(frame.sequenceID, sumPipeNanosElapsed, fps, targetList, frame);
    }
}
