This is an updated version of multired-0.1, as originally implemented by Vincenzo Nicosia in https://github.com/KatolaZ/multired

This code can be a valid alternative to using R packages for checking multiplex reducibility structure in networks with more than 10K nodes.

--- Upgrade List in Version 0.2

1) The syntax was updated from Python2 to Python3, making it possible to run the command on newer kernels.

2) Fixed a memory leak with multilayer networks having non-overlapping connected components.

3) Fixed a minor dependency issue so that multired 0.2 can now be used in Google Colab too (April 2025).

--- Usage

On Google Colab -> Upload multired2 in your local Runtime and then import it.
On VSC or similar environments -> Make sure to have multired2 in your local environment.
Remember that:
- there should be a file (e.g. edge_list.txt) containing only the files of the edge lists;
- edge lists should be in textual format, one row one edge in the format node_1 node_2, all nodes should be numerical (words are not supported), separators should be spaces (tabs not supported).

Usage is rather simple, double check the original commands from https://github.com/KatolaZ/multired or the following screenshot:

![immagine](https://github.com/user-attachments/assets/d0725bef-736b-4fd0-801f-9e13ef9d0f10)

Notice you can also run an approximate version of the analysis for eigenvalue estimations:

![immagine](https://github.com/user-attachments/assets/58a715e8-cbf9-4116-a977-aa017640dd80)



Important References:

This is a Python implmementation of the algorithm for structural reduction of multi-layer networks based on the Von Neumann Entropy and on the Quantum Jensen-Shannon divergence of graphs, as explained in:

M. De Domenico. V. Nicosia, A. Arenas, V. Latora "Structural reducibility of multilayer networks", Nat. Commun. 6, 6864 (2015) doi:10.1038/ncomms7864

Please cite their original paper.
