Models
======

Eight implementations, each on its own page: the problem it attacks, the method with its objective written out, the training loop step by step, a map from the method to the code, and the keys that run it.
Read :doc:`../concepts/select-then-predict` first if the architecture is new to you, since every page below names a difficulty stated there.

All eight answer one question, which is how to stop a selector and a predictor trained together from settling on an uninformative highlight they both agree about.

.. list-table::
   :header-rows: 1
   :widths: 12 46 22 20

   * - Model
     - What it changes
     - Venue
     - Cost
   * - :doc:`fr`
     - One encoder shared by selector and predictor, so neither can drift into a representation the other does not hold.
     - NeurIPS 2022
     - One encoder fewer
   * - :doc:`mgr`
     - Several generators against one shared predictor, so no single generator dictates the equilibrium.
     - ACL 2023
     - One encoder per generator
   * - :doc:`dr`
     - The predictor's learning rate scaled by what the selection kept, which restrains it while the selection is poor.
     - KDD 2023
     - None, it changes no loss
   * - :doc:`mcd`
     - A second prediction from the full input, and a highlight trained to make the two agree.
     - NeurIPS 2023
     - Two phases per batch
   * - :doc:`grat`
     - An attention classifier over the full input, supervising the selection and matched in distribution.
     - AAAI 2024
     - A third encoder, pretrained
   * - :doc:`dar`
     - A frozen aligner that only ever read full text, scoring the highlight it is handed.
     - ICDE 2024
     - A pretraining loop
   * - :doc:`mrd`
     - The question reversed: what the complement can still say, maximised rather than minimised.
     - NeurIPS 2024
     - Two phases per batch
   * - :doc:`genspp`
     - No gradient descent on the generator at all, and a genetic search in its place.
     - ACL 2025
     - A predictor per candidate

Writing a method of your own rather than running one of these is :doc:`../tutorials/custom-model`, which builds a small architecture end to end and names what to override for what.

Every architecture is registered for a GRU backbone and for a Transformer one.
The algorithms mention neither, since a backbone is anything implementing ``encode``, ``pool`` and ``output_size``, so swapping one for the other is a key rather than a code change.

.. toctree::
   :maxdepth: 1

   fr
   mgr
   dr
   mcd
   grat
   dar
   mrd
   genspp

Reading order
-------------

The order above is chronological, and it is also the order the pages read best in.
:doc:`fr` is the shortest intervention and the best place to see the base architecture with nothing added to it, while :doc:`mgr` and :doc:`dr` change the optimizer rather than the objective and are the two cheapest departures from it.
The phased pair :doc:`mcd` and :doc:`mrd` are worth reading together, since the second reverses the question the first asks and shares its training loop.
:doc:`genspp` is last because it abandons the assumption every other page shares, which is that the generator is trained by gradient descent at all.
