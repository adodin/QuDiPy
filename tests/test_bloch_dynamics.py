import unittest
from unittest import TestCase
import QuDiPy.math.linear_algebra as la
from QuDiPy.dynamics.unitary_dynamics import unitary_derivative, array_unitary_derivative, grid_unitary_derivative
from QuDiPy.dynamics.bloch_dynamics import bloch_derivative, array_bloch_derivative, grid_bloch_derivative
import QuDiPy.containers.grids as gr
import numpy as np


class TestBlochDerivative(TestCase):
    def test_bloch_derivative_unitary_consistency(self):
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rho = la.CartesianSpinOperator((0.5, 0, 0.5, 0))
        ud = unitary_derivative(rho, ham)
        bd = bloch_derivative(rho, ham, gdiss=0, gdeph=0)
        self.assertEqual(ud, bd)

    def test_bloch_derivative_no_unitary_part(self):
        ham = la.CartesianSpinOperator((0, 0, 0, 0))
        rho = la.CartesianSpinOperator((0.5, 0, 0.5, 0))
        bd_c = bloch_derivative(rho, ham, gdiss=0., gdeph=1.)
        bd_e = -1 * la.CartesianSpinOperator((0, 0, 0.5, 0))
        self.assertEqual(bd_c, bd_e)

    def test_bloch_derivative_all(self):
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rho = la.CartesianSpinOperator((0.5, 0, 0.5, 0))
        bd_c = bloch_derivative(rho, ham, gdiss=0., gdeph=1.)
        ud = unitary_derivative(rho, ham)
        nud = -1 * la.CartesianSpinOperator((0, 0, 0.5, 0))
        bd_e = ud + nud
        self.assertEqual(bd_c, bd_e)


class TestArrayBlochDerivative(TestCase):
    def test_bloch_derivative_all(self):
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rhoi = la.CartesianSpinOperator((0.5, 0., 0., 0.))
        rhox = la.CartesianSpinOperator((0.5, 0.5, 0., 0.))
        rhoy = la.CartesianSpinOperator((0.5, 0., 0.5, 0.))
        rhoz = la.CartesianSpinOperator((0.5, 0., 0., 0.5))
        rhos = np.array([rhoi, rhox, rhoy, rhoz])
        uds = array_unitary_derivative(rhos, ham)
        nuds = np.array([la.CartesianSpinOperator((0., 0., 0., 0.)), la.CartesianSpinOperator((0., -0.5, 0., 0.)),
                         la.CartesianSpinOperator((0., 0., -0.5, 0.)), la.CartesianSpinOperator((0., 0., 0., 0.))])
        bd_es = uds + nuds
        bd_cs = array_bloch_derivative(rhos, ham, gdiss=0., gdeph=1.)
        self.assertTrue(np.array_equal(bd_cs, bd_es))


class TestGridBlochDerivative(TestCase):
    def test_bloch_derivative_all(self):
        x1 = (1., 2.)
        x2 = (10., 20.)
        grid = gr.CartesianGrid((x1, x2))
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rhoi = la.CartesianSpinOperator((0.5, 0., 0., 0.))
        rhox = la.CartesianSpinOperator((0.5, 0.5, 0., 0.))
        rhoy = la.CartesianSpinOperator((0.5, 0., 0.5, 0.))
        rhoz = la.CartesianSpinOperator((0.5, 0., 0., 0.5))
        rhos = np.array([[rhoi, rhox], [rhoy, rhoz]])
        rho_grid = gr.GridData(rhos, grid)
        uds = grid_unitary_derivative(rho_grid, ham)
        nuds = rho_grid.like(np.array([[la.CartesianSpinOperator((0., 0., 0., 0.)),
                                        la.CartesianSpinOperator((0., -0.5, 0., 0.))],
                                       [la.CartesianSpinOperator((0., 0., -0.5, 0.)),
                                        la.CartesianSpinOperator((0., 0., 0., 0.))]]))
        bd_es = uds + nuds
        bd_cs = grid_bloch_derivative(rho_grid, ham, gdiss=0., gdeph=1.)
        self.assertTrue(np.array_equal(bd_cs, bd_es))


if __name__ == '__main__':
    unittest.main()
