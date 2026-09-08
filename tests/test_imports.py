def test_model_modules_import():
    from pyhighlights.components.models import InputData, OutputData
    from pyhighlights.components.models.spp import (
        FR,
        GRAT,
        MCD,
        MGR,
        GRUBackbone,
        SPPOutput,
        TransformerBackbone,
    )

    assert all(
        value is not None
        for value in (
            InputData,
            OutputData,
            SPPOutput,
            FR,
            MGR,
            MCD,
            GRAT,
            GRUBackbone,
            TransformerBackbone,
        )
    )
