.. _dev:

Development
===========

DeeR is a work in progress and contributions are welcome via pull request.

For more information, you can check out this link : |how_to_contrib|.

.. |how_to_contrib| raw:: html

   <a href="https://guides.github.com/activities/contributing-to-open-source/#contributing" target="_blank">Contributing to an open source Project on github</a>


You should also make sure that you install the repository approriately for development (see :ref:`dev-install`).

Guidelines for this project
---------------------------

Here are a few guidelines for this project.

* Simplicity: Be easy to use but also easy to understand when one digs into the code. Any additional code should be justified by the usefulness of the feature.
* Modularity: The user should be able to easily use its own code with any part of the deer framework (probably at the exception of the core of agent.py that is coded in a very general way).

These guidelines come of course in addition to all good practices for open source development.

.. _naming_conv:

Naming convention for this project
----------------------------------

* All classes and methods have word boundaries using medial capitalization. Classes are written with UpperCamelCase and methods are written with lowerCamelCase respectively. Example: "two words" is rendered as "TwoWords" for the UpperCamelCase (classes) and "twoWords" for the lowerCamelCase (methods).
* All attributes and variables have words separated by underscores. Example: "two words" is rendered as "two_words"
* If a variable is intended to be 'private', it is prefixed by an underscore.

