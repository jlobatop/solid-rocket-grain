solid-rocket-grain
===================

.. |triki| image:: https://media.giphy.com/media/o5oLImoQgGsKY/giphy.gif

Analysis of solid rocket grain distribution with celllular automata. 

**(... in progress)**

|triki|

####################
polarDiscretization
####################
This first approach was incorrect given that the polar grid did not respect the normal direction of the burning front (as it can be seen in the below picture of the star distribution burning).

.. image:: https://raw.githubusercontent.com/jlobatop/solid-rocket-grain/master/polarDiscretization/examples/cross.gif
	:width: 500pt
	:align: center

################
circularMapping
################
Completely new approach with a mapping of a circular grid from a squared 2D array. 