#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  multired.py (Python 3.7+ version)
#
#  Copyright (C) 2015 Vincenzo (Enzo) Nicosia
#  <katolaz@yahoo.it>
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License,
#  or (at your option) any later version.  
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#  See the GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#  This module provides the class multiplex_red, which implements the
#  algorithm for the structural reduction of multi-layer networks
#  based on the Von Neumann entropy and Quantum Jensen-Shannon
#  divergence of graphs. 
#
#  If you use this code please cite:
#    M. De Domenico, V. Nicosia, A. Arenas, V. Latora, 
#    "Structural reducibility of multilayer networks" 
#    Nat. Commun. 6, 6864 (2015) doi:10.1038/ncomms7864
#
#  -- Original version (Python 2.7)
#  -- This version updated for Python 3.7+

import sys
import math
import copy
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.linalg import eigh, eig
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

has_matplotlib = False
try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


class XLogx_fit:
    def __init__(self, degree, npoints=100, xmax=1):
        if xmax > 1:
            xmax = 1
        self.degree = degree
        x = np.linspace(0, xmax, npoints)
        # Compute y = x*log(x), handling x=0 carefully
        y = [val * math.log(val) if val > 0 else 0 for val in x]
        self.fit = np.polyfit(x, y, degree)

    def __getitem__(self, index):
        if index <= self.degree:
            return self.fit[index]
        else:
            print(f"Error!!! Index {index} is larger than the degree of the fitting polynomial ({self.degree})")
            sys.exit(-1)


class layer:
    def __init__(self, layerfile=None, matrix=None):
        self.N = 0
        self.num_layer = -1
        self.fname = layerfile
        self.adj_matr = None
        self.laplacian = None
        self.resc_laplacian = None
        self.entropy = None
        self.entropy_approx = None
        self._ii = []
        self._jj = []
        self._ww = []
        self._matrix_called = False

        if layerfile is not None:
            try:
                min_N = float('inf')
                with open(layerfile, "r") as lines:
                    for l in lines:
                        if l.startswith('#'):
                            continue
                        elems = l.strip(" \n").split(" ")
                        s = int(elems[0])
                        d = int(elems[1])
                        self._ii.append(s)
                        self._jj.append(d)
                        if s < min_N:
                            min_N = s
                        if d < min_N:
                            min_N = d
                        # Weight if provided
                        if len(elems) > 2:
                            val = [float(x) if "e" in x or "." in x else int(x)
                                   for x in [elems[2]]][0]
                            self._ww.append(float(val))
                        else:
                            self._ww.append(1)

                m1 = max(self._ii)
                m2 = max(self._jj)
                self.N = m1 if m1 > m2 else m2

            except IOError as e:
                print(f"Unable to find/open file {layerfile} -- Exiting!!!")
                sys.exit(-2)

        elif matrix is not None:
            self.adj_matr = copy.copy(matrix)
            self.N, _ = matrix.shape
            # Build Laplacian
            K = self.adj_matr.sum(0).reshape((1, self.N)).tolist()[0]
            D = csr_matrix((K, (range(self.N), range(self.N))), shape=(self.N, self.N))
            self.laplacian = csr_matrix(D - self.adj_matr)
            K_diag = self.laplacian.diagonal().sum()
            self.resc_laplacian = csr_matrix(self.laplacian / K_diag)
            self._matrix_called = True
        else:
            print("The given matrix is BLANK")

    def make_matrices(self, N):
        self.N = N
        self.adj_matr = csr_matrix((self._ww, (self._ii, self._jj)), shape=(self.N, self.N))
        self.adj_matr = self.adj_matr + self.adj_matr.transpose()
        K = self.adj_matr.sum(0).reshape((1, self.N)).tolist()[0]
        D = csr_matrix((K, (range(self.N), range(self.N))), shape=(self.N, self.N))
        self.laplacian = csr_matrix(D - self.adj_matr)
        K_diag = self.laplacian.diagonal().sum()
        self.resc_laplacian = csr_matrix(self.laplacian / K_diag)
        self._matrix_called = True

    def dump_info(self):
        N, _ = self.adj_matr.shape
        K = self.adj_matr.nnz
        sys.stderr.write(
            f"Layer File: {self.fname}\nNodes: {N} Edges: {K}\nEntropy: {self.entropy} "
            f"Approx. Entropy: {self.entropy_approx}\n"
        )

    def compute_VN_entropy(self):
        eigvals = eigh(self.resc_laplacian.todense())
        self.entropy = 0
        for l_i in eigvals[0]:
            if l_i > 1e-20:
                self.entropy -= l_i * math.log(l_i)

    def compute_VN_entropy_approx(self, poly):
        p = poly.degree
        h = -poly[p] * self.N
        M = csr_matrix(eye(self.N))
        for i in range(p - 1, -1, -1):
            M = M * self.resc_laplacian
            h += -poly[i] * sum(M.diagonal())
        self.entropy_approx = h

    def aggregate(self, other_layer):
        if self.adj_matr is not None:
            self.adj_matr = self.adj_matr + other_layer.adj_matr
        else:
            self.adj_matr = copy.copy(other_layer.adj_matr)
        K = self.adj_matr.sum(0).reshape((1, self.N)).tolist()[0]
        D = csr_matrix((K, (range(self.N), range(self.N))), shape=(self.N, self.N))
        self.laplacian = csr_matrix(D - self.adj_matr)
        K_diag = self.laplacian.diagonal().sum()
        self.resc_laplacian = csr_matrix(self.laplacian / K_diag)
        self._matrix_called = True

    def dump_laplacian(self):
        print(self.laplacian)


class multiplex_red:
    def __init__(self, multiplexfile, directed=None, fit_degree=10, verbose=False):
        self.layers = []
        self.N = 0
        self.M = 0
        self.entropy = 0
        self.entropy_approx = 0
        self.JSD = None
        self.JSD_approx = None
        self.Z = None
        self.Z_approx = None
        self.aggr = None
        self.q_vals = None
        self.q_vals_approx = None
        self.fit_degree = fit_degree
        self.poly = XLogx_fit(self.fit_degree)
        self.verb = verbose
        self.cuts = None
        self.cuts_approx = None

        try:
            with open(multiplexfile, "r") as lines:
                for l in lines:
                    if self.verb:
                        sys.stderr.write(f"Loading layer {len(self.layers)} from file {l}")
                    A = layer(l.strip(" \n"))
                    self.layers.append(A)

            # Build adjacency/laplacian for each
            N = max(x.N for x in self.layers)
            self.N = N + 1
            n = 0
            for lyr in self.layers:
                lyr.make_matrices(self.N)
                lyr.num_layer = n
                n += 1
            self.M = len(self.layers)

        except IOError as e:
            print(f"Unable to find/open file {multiplexfile} -- Exiting!!!")
            sys.exit(-2)

    def dump_info(self):
        i = 0
        for l in self.layers:
            sys.stderr.write(f"--------\nLayer: {i}\n")
            l.dump_info()
            i += 1

    def compute_aggregated(self):
        self.aggr = copy.copy(self.layers[0])
        self.aggr.entropy = 0
        self.aggr.entropy_approx = 0
        for l in self.layers[1:]:
            self.aggr.aggregate(l)

    def compute_layer_entropies(self):
        for l in self.layers:
            l.compute_VN_entropy()

    def compute_layer_entropies_approx(self):
        for l in self.layers:
            l.compute_VN_entropy_approx(self.poly)

    def compute_multiplex_entropy(self, force_compute=False):
        # Sum of the entropies of its layers
        for l in self.layers:
            if l.entropy is None:
                l.compute_VN_entropy()
            self.entropy += l.entropy

    def compute_multiplex_entropy_approx(self, force_compute=False):
        # Sum of the entropies of its layers
        for l in self.layers:
            if l.entropy_approx is None:
                l.compute_VN_entropy_approx(self.poly)
            self.entropy_approx += l.entropy_approx

    def compute_JSD_matrix(self):
        if self.verb:
            sys.stderr.write("Computing JSD matrix\n")
        self.JSD = np.zeros((self.M, self.M))
        for i in range(len(self.layers)):
            for j in range(i + 1, len(self.layers)):
                li = self.layers[i]
                lj = self.layers[j]
                if li.entropy is None:
                    li.compute_VN_entropy()
                if lj.entropy is None:
                    lj.compute_VN_entropy()

                # Create the layer from average adjacency
                m_sigma_matr = (li.adj_matr + lj.adj_matr) / 2.0
                m_sigma = layer(matrix=m_sigma_matr)
                m_sigma.compute_VN_entropy()

                d = m_sigma.entropy - 0.5 * (li.entropy + lj.entropy)
                d = math.sqrt(d)
                self.JSD[i][j] = d
                self.JSD[j][i] = d

    def compute_JSD_matrix_approx(self):
        if self.verb:
            sys.stderr.write("Computing JSD matrix (approx)\n")
        self.JSD_approx = np.zeros((self.M, self.M))
        for i in range(len(self.layers)):
            for j in range(i + 1, len(self.layers)):
                li = self.layers[i]
                lj = self.layers[j]
                if li.entropy_approx is None:
                    li.compute_VN_entropy_approx(self.poly)
                if lj.entropy_approx is None:
                    lj.compute_VN_entropy_approx(self.poly)

                m_sigma_matr = (li.adj_matr + lj.adj_matr) / 2.0
                m_sigma = layer(matrix=m_sigma_matr)
                m_sigma.compute_VN_entropy_approx(self.poly)

                d = m_sigma.entropy_approx - 0.5 * (li.entropy_approx + lj.entropy_approx)
                d = math.sqrt(d)
                self.JSD_approx[i][j] = d
                self.JSD_approx[j][i] = d

    def dump_JSD(self, force_compute=False):
        if self.JSD is None:
            if force_compute:
                self.compute_JSD_matrix()
            else:
                print("Error!!! call to dump_JSD but JSD matrix has not been computed!!!")
                sys.exit(1)
        idx = 0
        for i in range(self.M):
            for j in range(i + 1, self.M):
                print(i, j, self.JSD[i][j])
                idx += 1

    def dump_JSD_approx(self, force_compute=False):
        if self.JSD_approx is None:
            if force_compute:
                self.compute_JSD_matrix_approx()
            else:
                print("Error!!! call to dump_JSD_approx but JSD approximate matrix has not been computed!!!")
                sys.exit(1)
        idx = 0
        for i in range(self.M):
            for j in range(i + 1, self.M):
                print(i, j, self.JSD_approx[i][j])
                idx += 1

    def reduce(self, method="ward"):
        if self.verb:
            sys.stderr.write(f"Performing '{method}' reduction\n")
        if self.JSD is None:
            self.compute_JSD_matrix()
        self.Z = linkage(squareform(self.JSD, checks=False), method=method)
        return self.Z

    def reduce_approx(self, method="ward"):
        if self.verb:
            sys.stderr.write(f"Performing '{method}' reduction (approx)\n")
        if self.JSD_approx is None:
            self.compute_JSD_matrix_approx()
        # Convert full square matrix to condensed form
        condensed_JSD_approx = squareform(self.JSD_approx, checks=False)
        self.Z_approx = linkage(condensed_JSD_approx, method=method)
        
        return self.Z_approx

    def get_linkage(self):
        return self.Z

    def get_linkage_approx(self):
        return self.Z_approx

    def __compute_q(self, layers):
        if not self.aggr:
            self.compute_aggregated()
            self.aggr.compute_VN_entropy()

        H_avg = 0
        for l in layers:
            if l.entropy is None:
                l.compute_VN_entropy()
            H_avg += l.entropy
        H_avg /= len(layers)

        q = 1.0 - H_avg / self.aggr.entropy
        return q

    def get_q_profile(self):
        mylayers = copy.copy(self.layers)
        rem_layers = copy.copy(self.layers)
        q_vals = []
        if self.Z is None:
            self.reduce()
        q = self.__compute_q(rem_layers)
        q_vals.append(q)
        n = len(self.layers)

        for l1, l2, _d, _x in self.Z:
            l1 = int(l1)
            l2 = int(l2)
            new_layer = layer(matrix=mylayers[l1].adj_matr)
            new_layer.num_layer = n
            n += 1
            new_layer.aggregate(mylayers[l2])
            rem_layers.remove(mylayers[l1])
            rem_layers.remove(mylayers[l2])
            rem_layers.append(new_layer)
            mylayers.append(new_layer)
            q = self.__compute_q(rem_layers)
            q_vals.append(q)

        self.q_vals = q_vals
        return q_vals

    def __compute_q_approx(self, layers):
        if not self.aggr:
            self.compute_aggregated()
            self.aggr.compute_VN_entropy_approx(self.poly)

        H_avg = 0
        for l in layers:
            if l.entropy_approx is None:
                l.compute_VN_entropy_approx(self.poly)
            H_avg += l.entropy_approx
        H_avg /= len(layers)

        q = 1.0 - H_avg / self.aggr.entropy_approx
        return q

    def get_q_profile_approx(self):
        mylayers = copy.copy(self.layers)
        rem_layers = copy.copy(self.layers)
        q_vals = []
        if self.Z_approx is None:
            self.reduce_approx()
        q = self.__compute_q_approx(rem_layers)
        q_vals.append(q)
        n = len(self.layers)

        for l1, l2, _d, _x in self.Z_approx:
            l1 = int(l1)
            l2 = int(l2)
            new_layer = layer(matrix=mylayers[l1].adj_matr)
            new_layer.num_layer = n
            n += 1
            new_layer.aggregate(mylayers[l2])
            rem_layers.remove(mylayers[l1])
            rem_layers.remove(mylayers[l2])
            rem_layers.append(new_layer)
            mylayers.append(new_layer)
            q = self.__compute_q_approx(rem_layers)
            q_vals.append(q)

        self.q_vals_approx = q_vals
        return q_vals

    def compute_partitions(self):
        if self.verb:
            sys.stderr.write("Getting partitions...\n")
        if self.Z is None:
            self.reduce()
        if self.q_vals is None:
            self.get_q_profile()

        sets_map = {}
        M = len(self.layers)
        for i in range(len(self.layers)):
            sets_map[i] = [i]

        cur_part = list(sets_map.values())
        self.cuts = [copy.deepcopy(cur_part)]
        j = 0

        while j < M - 1:
            l1, l2, _x, _y = self.Z[j]
            l1 = int(l1)
            l2 = int(l2)
            val = sets_map[l1]
            val.extend(sets_map[l2])
            sets_map[M + j] = val
            cur_part.remove(sets_map[l1])
            cur_part.remove(sets_map[l2])
            cur_part.append(val)
            j += 1
            self.cuts.append(copy.deepcopy(cur_part))

        self.cuts.append(copy.deepcopy(cur_part))
        return list(zip(self.q_vals, self.cuts))

    def compute_partitions_approx(self):
        if self.verb:
            sys.stderr.write("Getting partitions (approx)...\n")
        if self.Z_approx is None:
            self.reduce_approx()
        if self.q_vals_approx is None:
            self.get_q_profile_approx()

        sets_map = {}
        M = len(self.layers)
        for i in range(len(self.layers)):
            sets_map[i] = [i]

        cur_part = list(sets_map.values())
        self.cuts_approx = [copy.deepcopy(cur_part)]
        j = 0

        while j < M - 1:
            l1, l2, _x, _y = self.Z_approx[j]
            l1 = int(l1)
            l2 = int(l2)
            val = sets_map[l1]
            val.extend(sets_map[l2])
            sets_map[M + j] = val
            cur_part.remove(sets_map[l1])
            cur_part.remove(sets_map[l2])
            cur_part.append(val)
            j += 1
            self.cuts_approx.append(copy.deepcopy(cur_part))

        self.cuts_approx.append(copy.deepcopy(cur_part))
        return list(zip(self.q_vals_approx, self.cuts_approx))

    def draw_dendrogram(self, force=False):
        if not has_matplotlib:
            sys.stderr.write("No matplotlib module found in draw_dendrogram...Exiting!!!\n")
            sys.exit(3)
        if self.Z is None:
            if not force:
                sys.stderr.write("Please call reduce() first or specify 'force=True'")
            else:
                self.reduce()
        dendrogram(self.Z, no_plot=False)
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    def draw_dendrogram_approx(self, force=False):
        if not has_matplotlib:
            sys.stderr.write("No matplotlib module found in draw_dendrogram_approx...Exiting!!!\n")
            sys.exit(3)
        if self.Z_approx is None:
            if not force:
                sys.stderr.write("Please call reduce_approx() first or specify 'force=True'")
            else:
                self.reduce_approx()
        dendrogram(self.Z_approx, no_plot=False)
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    def dump_partitions(self):
        part = list(zip(self.q_vals, self.cuts))
        for q, p in part:
            print(q, "->", p)

    def dump_partitions_approx(self):
        part = list(zip(self.q_vals_approx, self.cuts_approx))
        for q, p in part:
            print(q, "->", p)
