import unittest
import numpy as np
from unittest import TestCase
import QuDiPy.math.linear_algebra as la
from QuDiPy.util.constants import hbar
import QuDiPy.containers.grids as gr
from QuDiPy.dynamics.unitary_dynamics import unitary_derivative, array_unitary_derivative, grid_unitary_derivative


class TestUnitaryDerivative(TestCase):
    def test_steady_state_unitary_dynamics_cartesian_spin_operator(self):
        rho0 = la.CartesianSpinOperator((1/2, 0, 0, 0))
        rhodot_expected = la.CartesianSpinOperator((0, 0, 0, 0))
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        self.assertEqual(unitary_derivative(rho0, ham), rhodot_expected)

    def test_steady_state_unitary_dynamics_spherical_spin_operator(self):
        rho0 = la.SphericalSpinOperator((1 / 2, 0, 0, 0))
        rhodot_expected = la.SphericalSpinOperator((0, 0, 0, 0))
        ham = la.SphericalSpinOperator((0, 0, 0, 1))
        self.assertEqual(unitary_derivative(rho0, ham), rhodot_expected)

    def test_steady_state_unitary_dynamics_mixed_spin_operator(self):
        rho0 = la.SphericalSpinOperator((1 / 2, 0, 0, 0))
        rhodot_expected = la.SphericalSpinOperator((0, 0, 0, 0))
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        self.assertEqual(unitary_derivative(rho0, ham), rhodot_expected)

    def test_steady_state_unitary_dynamics_numpy_array(self):
        rho0 = np.array([[0.5, 0.], [0., 0.5]])
        rhodot_expected = np.array([[0., 0.], [0., 0.]])
        ham = np.array([[1., 0.], [0., -1.]])
        self.assertTrue(np.array_equal(unitary_derivative(rho0, ham), rhodot_expected))

    def test_nonzero_unitary_dynamics_cartesian_spin_operator(self):
        rhox = la.CartesianSpinOperator((0.5, 0.5, 0., 0.))
        rhoy = la.CartesianSpinOperator((0.5, 0., 0.5, 0.))
        rhoxy = la.CartesianSpinOperator((0.5, 0.5, 0.5, 0.))
        rhoxdot_expected = 1 / hbar * la.CartesianSpinOperator((0., 0., 1., 0.))
        rhoydot_expected = 1 / hbar * la.CartesianSpinOperator((0., -1., 0., 0.))
        rhoxydot_expected = 1 / hbar * la.CartesianSpinOperator((0., -1., 1., 0.))
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rhoxdot_calc = unitary_derivative(rhox, ham)
        rhoydot_calc = unitary_derivative(rhoy, ham)
        rhoxydot_calc = unitary_derivative(rhoxy, ham)
        self.assertEqual(rhoxdot_calc, rhoxdot_expected)
        self.assertEqual(rhoydot_calc, rhoydot_expected)
        self.assertEqual(rhoxydot_calc, rhoxydot_expected)

    def test_nonzero_unitary_dynamics_spherical_spin_operator(self):
        rhox = la.SphericalSpinOperator((0.5, 0.5, np.pi/2, 0.))
        rhoy = la.SphericalSpinOperator((0.5, 0.5, np.pi/2, np.pi/2))
        rhoxy = la.SphericalSpinOperator((0.5, 0.5, np.pi/2, np.pi/4))
        rhoxdot_expected = 1 / hbar * la.SphericalSpinOperator((0., 1., np.pi/2, np.pi/2))
        rhoydot_expected = 1 / hbar * la.SphericalSpinOperator((0., 1., np.pi/2, np.pi))
        rhoxydot_expected = 1 / hbar * la.SphericalSpinOperator((0., 1., np.pi/2, 3*np.pi/4))
        ham = la.SphericalSpinOperator((0, 1, 0, 0))
        rhoxdot_calc = unitary_derivative(rhox, ham)
        rhoydot_calc = unitary_derivative(rhoy, ham)
        rhoxydot_calc = unitary_derivative(rhoxy, ham)
        self.assertEqual(rhoxdot_calc, rhoxdot_expected)
        self.assertEqual(rhoydot_calc, rhoydot_expected)
        self.assertEqual(rhoxydot_calc, rhoxydot_expected)

    def test_nonzero_unitary_dynamics_mixed_spin_operator(self):
        rhox = la.CartesianSpinOperator((0.5, 0.5, 0., 0.))
        rhoy = la.CartesianSpinOperator((0.5, 0., 0.5, 0.))
        rhoxy = la.CartesianSpinOperator((0.5, 0.5, 0.5, 0.))
        rhoxdot_expected = la.CartesianSpinOperator((0., 0., 1., 0.))
        rhoydot_expected = la.CartesianSpinOperator((0., -1., 0., 0.))
        rhoxydot_expected = la.CartesianSpinOperator((0., -1., 1., 0.))
        ham = la.SphericalSpinOperator((0, 1, 0, 0))
        rhoxdot_calc = hbar * unitary_derivative(rhox, ham)
        rhoydot_calc = hbar * unitary_derivative(rhoy, ham)
        rhoxydot_calc = hbar * unitary_derivative(rhoxy, ham)
        self.assertAlmostEqual(rhoxdot_calc, rhoxdot_expected)
        self.assertAlmostEqual(rhoydot_calc, rhoydot_expected)
        self.assertAlmostEqual(rhoxydot_calc, rhoxydot_expected)


class TestArrayUnitaryDerivative(TestCase):
    def test_nonzero_unitary_dynamics(self):
        rhox = la.CartesianSpinOperator((0.5, 0.5, 0., 0.))
        rhoy = la.CartesianSpinOperator((0.5, 0., 0.5, 0.))
        rhoxy = la.CartesianSpinOperator((0.5, 0.5, 0.5, 0.))
        rhos = np.array([rhox, rhoy, rhoxy])
        rhoxdot_expected = 1 / hbar * la.CartesianSpinOperator((0., 0., 1., 0.))
        rhoydot_expected = 1 / hbar * la.CartesianSpinOperator((0., -1., 0., 0.))
        rhoxydot_expected = 1 / hbar * la.CartesianSpinOperator((0., -1., 1., 0.))
        rhodots_expected = np.array([rhoxdot_expected, rhoydot_expected, rhoxydot_expected])
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rhodots_calc = array_unitary_derivative(rhos, ham)
        self.assertTrue(np.array_equal(rhodots_expected, rhodots_calc))


class TestGridUnitaryDerivative(TestCase):
    def test_nonzero_unitary_dynamics(self):
        x1 = (1., 2.)
        x2 = (10., 20.)
        grid = gr.CartesianGrid((x1, x2))
        rhoi = la.CartesianSpinOperator((0.5, 0., 0., 0.))
        rhox = la.CartesianSpinOperator((0.5, 0.5, 0., 0.))
        rhoy = la.CartesianSpinOperator((0.5, 0., 0.5, 0.))
        rhoz = la.CartesianSpinOperator((0.5, 0., 0., 0.5))
        rhos = np.array([[rhoi, rhox], [rhoy, rhoz]])
        rho_grid = gr.DataGrid(rhos, grid)
        rhoidot_expected = 1 / hbar * la.CartesianSpinOperator((0., 0., 0., 0.))
        rhoxdot_expected = 1 / hbar * la.CartesianSpinOperator((0., 0., 1., 0.))
        rhoydot_expected = 1 / hbar * la.CartesianSpinOperator((0., -1., 0., 0.))
        rhozdot_expected = 1 / hbar * la.CartesianSpinOperator((0., 0., 0., 0.))
        rhodots_expected = np.array([[rhoidot_expected, rhoxdot_expected], [rhoydot_expected, rhozdot_expected]])
        rhodot_expected_grid = rho_grid.like(rhodots_expected)
        ham = la.CartesianSpinOperator((0, 0, 0, 1))
        rhodots_calc_grid = grid_unitary_derivative(rho_grid, ham)
        self.assertEqual(rhodot_expected_grid, rhodots_calc_grid)


if __name__ == '__main__':
    unittest.main()
