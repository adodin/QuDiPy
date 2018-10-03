import unittest
import numpy as np
from unittest import TestCase
import QuDiPy.util.linalg as la
from QuDiPy.util.constants import hbar
from QuDiPy.dynamics.unitary_dynamics import unitary_derivative


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
        self.assertEqual(rhoxdot_calc, rhoxdot_expected)
        self.assertEqual(rhoydot_calc, rhoydot_expected)
        self.assertEqual(rhoxydot_calc, rhoxydot_expected)


if __name__ == '__main__':
    unittest.main()
