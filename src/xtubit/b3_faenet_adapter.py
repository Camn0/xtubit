from __future__ import annotations


def build_faenet(cutoff=6.0, hidden_channels=128, num_filters=128, num_gaussians=50,
                 num_interactions=4):
    from faenet.model import FAENet
    return FAENet(
        cutoff=cutoff,
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_gaussians=num_gaussians,
        num_interactions=num_interactions,
        phys_embeds=True,
    )


def make_transform(frame_averaging="3D", fa_method="stochastic"):
    from faenet.transforms import FrameAveraging
    return FrameAveraging(frame_averaging, fa_method)


def forward_with_frames(batch, model, frame_averaging="3D", crystal_task=False, mode="inference"):
    from faenet.fa_forward import model_forward
    return model_forward(batch=batch, model=model,
                         frame_averaging=frame_averaging,
                         mode=mode, crystal_task=crystal_task)
