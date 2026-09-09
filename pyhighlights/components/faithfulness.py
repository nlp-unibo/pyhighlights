"""Faithfulness: whether the highlight is what the prediction rests on.

A model that reports a highlight makes a claim about itself, and metrics
against an annotation do not check it -- a highlight can match the annotation
and still not be what the predictor read. Two measures test the claim by
changing what the predictor sees, both from DeYoung, Jain, Rajani, Lehman,
Xiong, Socher and Wallace, 2020, *ERASER: A Benchmark to Evaluate Rationalized
NLP Models*.

Writing ``x`` for the input, ``h`` for the highlight, ``x \\ h`` for the input
with the highlight removed, and ``y_hat`` for the class scored:

.. code-block:: text

   sufficiency       = p(y_hat | x) - p(y_hat | h)
   comprehensiveness = p(y_hat | x) - p(y_hat | x \\ h)

Sufficiency asks whether the highlight carries the signal on its own, so
**lower is better**. Comprehensiveness asks whether anything the class rests
on was left outside the highlight, so **higher is better**. They are not two
views of one number: a model can highlight three words that suffice while ten
others would have sufficed too -- sufficient, and not comprehensive.

The literature calls ``h`` the *rationale* and writes it ``r``. This library
says highlight throughout, as its loaders, models and metrics do.

**Every architecture is measured partly off its training distribution, and
which part differs.** Nothing is trained on an input with a hole in it, so
comprehensiveness is off-distribution for everyone. The other two passes
divide by family:

=========================  ==============  ==============  ==================
Trained on                 ``p(y_hat|x)``  ``p(y_hat|h)``  ``p(y_hat|x \\ h)``
=========================  ==============  ==============  ==================
full input                 in              off             off
select-then-predict        off             in              off
MCD (both, by design)      in              in              off
=========================  ==============  ==============  ==================

So this is a property of the measure rather than a demerit of a model, and the
published numbers carry the same distortion with the terms exchanged.

Two consequences worth knowing before reading a column:

- **``y_hat`` is the class predicted from the highlight**, not from the full
  input. ERASER takes it from the full input because for a full-input
  classifier that *is* the model's prediction; the intent is the class the
  model predicts, and for a select-then-predict model that comes from ``h``.
  Anchoring on the full-input pass would anchor on the one pass such a model
  was never trained for. A documented deviation, and the reason a column here
  is not interchangeable with a published one.
- **Negative sufficiency is expected here.** ``p(y_hat | h)`` is the trained
  pass and ``p(y_hat | x)`` is not, so a select-then-predict model can score
  better on its highlight than on the whole input. That is not a defect.
"""

from __future__ import annotations

from typing import Dict

import torch as th
from torch.utils.data import DataLoader

from pyhighlights.components.models.base import Model

__all__ = ["evaluate"]


def evaluate(model: Model, loader: DataLoader) -> Dict[str, float]:
    """Mean faithfulness terms over ``loader``, as the model currently stands.

    Runs outside the Lightning loop: the terms need extra forward passes with
    masks of their own rather than another binding over the fields a step
    already produced. The model is left in the mode it arrived in.
    """
    totals: Dict[str, float] = {}
    count = 0

    training = model.training
    model.eval()
    try:
        with th.no_grad():
            for batch in loader:
                batch = type(batch)(
                    **{
                        name: value.to(model.device)
                        if isinstance(value, th.Tensor)
                        else value
                        for name, value in batch.as_dict().items()
                    }
                )
                terms = model.faithfulness(batch, model.test_forward(batch))
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.0) + float(value.sum())
                count += int(batch.mask.shape[0])
    finally:
        model.train(training)

    if not count:
        raise ValueError("faithfulness needs at least one example to score")
    return {name: value / count for name, value in totals.items()}
