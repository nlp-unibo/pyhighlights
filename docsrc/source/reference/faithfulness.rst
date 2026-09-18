Faithfulness
============

Metrics against an annotation check whether a highlight matches what a human marked.
They cannot check whether the highlight is what the predictor read, a model can match the annotation and rest its prediction on something else.
``faithfulness`` adds two measures that test the claim by changing what the predictor sees:

.. code-block:: python

   Registry.from_key(TOY_TASK, faithfulness=True).run()
   # results.json gains test_sufficiency and test_comprehensiveness

Writing ``x`` for the input, ``h`` for the highlight and ``x \ h`` for the input with the highlight removed:

.. code-block:: text

   sufficiency       = p(y_hat | x) - p(y_hat | h)
   comprehensiveness = p(y_hat | x) - p(y_hat | x \ h)

Sufficiency asks whether the highlight carries the signal alone, so **lower is better**.
Comprehensiveness asks whether anything the class rests on was left outside it, so **higher is better**.
Two extra predictor passes per test batch pay for both: the highlight pass is the model's own output, already computed.

Off by default.
They are two more columns rather than a correction, and a registered reproduction should report what its paper reports.

**Every architecture is scored partly off its training distribution, and which part differs.** Nothing is trained on an input with a hole in it, so comprehensiveness is off-distribution for everyone.
The other passes divide by family: a full-input classifier is trained for ``p(y_hat | x)`` and never sees ``h`` alone, while a select-then-predict predictor is trained for ``p(y_hat | h)`` and never sees ``x``.
MCD is the exception, it trains both passes by design.
So this is a property of the measure, not a demerit of a model, and published numbers carry the same distortion with the terms exchanged.

**Comprehensiveness can be uninformative rather than low.** Both of its terms are the full-length pass.
For a select-then-predict model that is the pass it never trains on, so both are degraded and their difference is compressed toward zero whatever the highlight contains.
Measured on a legal corpus, four words kept out of thirty-five:

.. code-block:: text

   cell            p(y_hat|h)   p(y_hat|x)   p(y_hat|x \ h)   comprehensiveness
   FR (GRU)             0.979        0.741           0.730              +0.012
   G-RAT (GRU)          0.982        0.764           0.761              +0.003
   MGR (GRU)            0.936        0.560           0.539              +0.020

The model answers 0.98 on the highlight and 0.74 on the whole clause.
Removing four words from thirty-five leaves thirty-one, which is the same off-training input, so the prediction does not move and the column reads near zero.
That is a fact about the measure applied to this model class, not a finding about the highlight.
**Do not read a low value here as a demerit** unless ``p(y_hat | x)`` is a pass the model is trained for.

Two consequences to know before reading a column:

* ``y_hat`` **is the class predicted from the highlight**, not from the full
  input. ERASER takes it from the full input because for a full-input
  classifier that *is* the model's prediction; the intent is the class the
  model predicts, and here that comes from ``h``. Anchoring on the full-input
  pass would anchor on the one pass such a model was never trained for. A
  documented deviation, and the reason a column here is not interchangeable
  with a published one.
* **Negative sufficiency is expected.** ``p(y_hat | h)`` is the trained pass
  and ``p(y_hat | x)`` is not, so a select-then-predict model can score better
  on its highlight than on the whole input. Not a defect.

The library says *highlight* where the literature says *rationale*, and writes ``h`` where it writes ``r``.
The metric names stay as published, so a column still matches a paper's.

API
---

.. automodule:: pyhighlights.components.faithfulness
   :members:
   :show-inheritance:
