import unittest
from unittest import TestCase
import QuDiPy.util.linalg as la
from QuDiPy.dynamics.unitary_dynamics import unitary_derivative
from QuDiPy.dynamics.bloch_dynamics import bloch_derivative


class TestBlochDerivative(TestCase):
    def test_bloch_derivative_unitary_consistency(self):
        ham = la.CartesianSpinOperator((0, 1, 0, 0))
        rho = la.CartesianSpinOperator((0, 0, 1, 0))
        ud = unitary_derivative(rho, ham)
        bd = bloch_derivative(rho, ham, gdiss=0, gdeph=0)
        self.assertEqual(ud, bd)

    def test_bloch_derivative_no_unitary_part(self):
        ham = la.CartesianSpinOperator((0, 0, 0, 0))
        rho = la.CartesianSpinOperator((0, 0, 1, 0))
        bd_c = bloch_derivative(rho, ham, gdiss=0., gdeph=1.)
        bd_e = -1 * la.CartesianSpinOperator((0, 0, 1, 0))
        self.assertEqual(bd_c, bd_e)


if __name__ == '__main__':
    unittest.main()
