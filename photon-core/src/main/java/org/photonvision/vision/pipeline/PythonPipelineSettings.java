package org.photonvision.vision.pipeline;

import com.fasterxml.jackson.annotation.JsonTypeName;
import java.util.Objects;

@JsonTypeName("PythonPipelineSettings")
public class PythonPipelineSettings extends AdvancedPipelineSettings{
    
    public PythonPipelineSettings() {
        super();
        pipelineType = PipelineType.Python;
        cameraExposureRaw = 20;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        PythonPipelineSettings that = (PythonPipelineSettings) o;
        return true;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode());
    }
}
