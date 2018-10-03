from unittest import TestCase
import unittest
import QuDiPy.util.linalg as la
import numpy.random as rn
import numpy as np

# Initialize the Random Seed (Using Date of Test Writing)
rn.seed(91818)


class TestCartesianSpinObservable(TestCase):
    """ This test class also tests the general functions and overloads of the Observable Class"""
    def setUp(self):
        """ Generates Basis Observables and a list of 10 random matrices for testing"""
        self.i = (1., 0., 0., 0.)
        self.x = (0., 1., 0., 0.)
        self.y = (0., 0., 1., 0.)
        self.z = (0., 0., 0., 1.)
        self.basis_vectors = [self.i, self.x, self.y, self.z]
        self.rands = [(rn.rand(), rn.rand(), rn.rand(), rn.rand()) for i in range(10)]
        self.vectors = self.basis_vectors + self.rands
        self.si = la.CartesianSpinOperator(self.i)
        self.si2 = la.CartesianSpinOperator(self.i)
        self.sx = la.CartesianSpinOperator(self.x)
        self.sy = la.CartesianSpinOperator(self.y)
        self.sz = la.CartesianSpinOperator(self.z)
        self.basis_operators = [self.si, self.sx, self.sy, self.sz]
        self.srands = [la.CartesianSpinOperator(rand) for rand in self.rands]
        self.operators = self.basis_operators + self.srands
        pass

    def test_init(self):
        """ Tests that all constructors initialize the correct Observables"""
        for op, vec in zip(self.basis_operators, self.vectors):
            self.assertEqual(op.vector, vec)

    def test_init_length_assertion(self):
        """ Tests that the constructor correctly raises initialization error if the wrong length input provided """
        with self.assertRaises(AssertionError):
            la.CartesianSpinOperator((1., 0., 0.))
        with self.assertRaises(AssertionError):
            la.CartesianSpinOperator((1., 0., 0., 0., 0., 0.))

    # Observable Overload Test
    def test_mult_overload(self):
        """ Tests that multiplication overload correctly calls the dot method """
        for ss in self.operators:
            for tt in self.operators:
                self.assertEqual(ss * tt, ss.dot(tt))

    # Observable Overload Test
    def test_abs_overload(self):
        """ Tests that the abs overload correctly calls the dot method """
        for ss in self.operators:
            self.assertEqual(abs(ss), np.sqrt(ss * ss))

    # Observable Overload Test
    def test_eq_overload(self):
        """ Tests that the == Observable was correctly overloaded"""
        self.assertTrue(self.si == self.si)
        self.assertTrue(self.si == self.si2)
        self.assertFalse(self.si == self.sx)
        self.assertTrue(self.si != self.sx)

    def test_matmul_overload(self):
        prodiz = self.si @ self.sz
        self.assertEqual(prodiz, self.sz)
        prodxz = self.sx @ self.sz
        prodzx = self.sz @ self.sx
        self.assertEqual(prodxz, la.CartesianSpinOperator((0., 0., -1.j, 0.)))
        self.assertEqual(prodzx, la.CartesianSpinOperator((0., 0., 1.j, 0.)))

    def test_add_overload(self):
        sum_ix = self.si + self.sx
        expected_sum_ix = la.CartesianSpinOperator((1, 1, 0, 0))
        self.assertEqual(sum_ix, expected_sum_ix)

    def test_rmult_real_overload(self):
        mult_2i = 2 * self.si
        expected_mult_2i = la.CartesianSpinOperator((2, 0, 0, 0))
        self.assertEqual(mult_2i, expected_mult_2i)

    def test_rmult_complex_overload(self):
        calc = (1 + 1j) * self.sx
        expected = la.CartesianSpinOperator((0, 1 + 1j, 0, 0))
        self.assertEqual(calc, expected)

    def test_sub_overload(self):
        diff_ix = self.si - self.sx
        expected_diff_ix = la.CartesianSpinOperator((1, -1, 0, 0))
        self.assertEqual(diff_ix, expected_diff_ix)

    def test_calculate_matrix(self):
        """ Tests that matrices are correctly constructed from the Observable """
        si_mat = np.array([[1., 0.], [0., 1.]])
        sx_mat = np.array([[0., 1.], [1., 0.]])
        sy_mat = np.array([[0., -1.j], [1.j, 0.]])
        sz_mat = np.array([[1., 0.], [0., -1.]])
        self.assertTrue(np.array_equal(np.round(self.si.calculate_matrix(), 7), si_mat))
        self.assertTrue(np.array_equal(np.round(self.sx.calculate_matrix(), 7), sx_mat))
        self.assertTrue(np.array_equal(np.round(self.sy.calculate_matrix(), 7), sy_mat))
        self.assertTrue(np.array_equal(np.round(self.sz.calculate_matrix(), 7), sz_mat))

    def test_basis_normalization(self):
        """ Tests that normalization of unit vectors is correct for basis Observables """
        self.assertAlmostEqual(abs(self.si), 1.0)
        self.assertAlmostEqual(abs(self.sx), 1.0)
        self.assertAlmostEqual(abs(self.sy), 1.0)
        self.assertAlmostEqual(abs(self.sz), 1.0)

    def test_basis_orthogonality(self):
        """ Tests that the dot product correctly predicts orthonormal bases"""
        self.assertAlmostEqual(self.si * self.sx, 0.)
        self.assertAlmostEqual(self.si * self.sy, 0.)
        self.assertAlmostEqual(self.si * self.sz, 0.)
        self.assertAlmostEqual(self.sx * self.sy, 0.)
        self.assertAlmostEqual(self.sx * self.sz, 0.)
        self.assertAlmostEqual(self.sy * self.sz, 0.)

    def test_convert_cartesian(self):
        """ Tests Cartesian Conversion of the Matrix """
        self.assertEqual(self.si.convert_cartesian(), self.i)
        self.assertEqual(self.sx.convert_cartesian(), self.x)
        self.assertEqual(self.sy.convert_cartesian(), self.y)
        self.assertEqual(self.sz.convert_cartesian(), self.z)
        for ss, vv in zip(self.srands, self.rands):
            self.assertEqual(ss.convert_cartesian(), vv)


class TestSphericalSpinObservable(TestCase):
    def setUp(self):
        self.i = (1., 0., 0., 0.)
        self.x = (0., 1., np.pi/2, 0.)
        self.y = (0., 1., np.pi/2, np.pi/2)
        self.z = (0., 1., 0., 0.)
        self.si = la.SphericalSpinOperator(self.i)
        self.sx = la.SphericalSpinOperator(self.x)
        self.sy = la.SphericalSpinOperator(self.y)
        self.sz = la.SphericalSpinOperator(self.z)
        pass

    def test_init(self):
        """ Tests the initialization of """
        self.assertEqual(self.si.vector, self.i)
        self.assertEqual(self.sx.vector, self.x)
        self.assertEqual(self.sy.vector, self.y)
        self.assertEqual(self.sz.vector, self.z)

    def test_matmul(self):
        prodiz = self.si @ self.sz
        self.assertEqual(prodiz, self.sz)
        prodxz = self.sx @ self.sz
        self.assertEqual(prodxz, -1j * self.sy)

    def test_cross_equality(self):
        cart_x = la.CartesianSpinOperator((0, 1, 0, 0))
        self.assertEqual(self.sx, cart_x)

    def test_add_overload(self):
        calc = self.si + self.sx
        expected = la.CartesianSpinOperator((1, 1, 0, 0))
        self.assertEqual(calc, expected)

    def test_rmult_real_overload(self):
        calc = -2 * self.sx
        expected = la.CartesianSpinOperator((0, -2, 0, 0))
        self.assertEqual(calc, expected)

    def test_rmult_complex_overload(self):
        calc = (1 + 1j) * self.sx
        expected = la.CartesianSpinOperator((0, 1 + 1j, 0, 0))
        self.assertEqual(calc, expected)

    def test_sub_overload(self):
        calc = self.si - self.sx
        expected = la.CartesianSpinOperator((1, -1, 0, 0))
        self.assertEqual(calc, expected)

    def test_init_length_assertion(self):
        """ Tests that the constructor correctly raises initialization error if the wrong length input provided """
        with self.assertRaises(AssertionError):
            la.SphericalSpinOperator((1., 0., 0.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinOperator((1., 0., 0., 0., 0., 0.))

    def test_convert_cartesian(self):
        """ Tests conversion between Spherical and Cartesian Coordinates """
        cart_i = self.i
        cart_x = (0., 1., 0., 0.)
        cart_y = (0., 0., 1., 0.)
        cart_z = (0., 0., 0., 1.)
        i_convert = self.si.convert_cartesian()
        x_convert = self.sx.convert_cartesian()
        y_convert = self.sy.convert_cartesian()
        z_convert = self.sz.convert_cartesian()
        for conv, cart in zip(i_convert, cart_i):
            self.assertAlmostEqual(conv, cart)
        for conv, cart in zip(x_convert, cart_x):
            self.assertAlmostEqual(conv, cart)
        for conv, cart in zip(y_convert, cart_y):
            self.assertAlmostEqual(conv, cart)
        for conv, cart in zip(z_convert, cart_z):
            self.assertAlmostEqual(conv, cart)

    def test_calculate_matrix(self):
        """ Tests Matrix Calculation """
        si_mat = np.array([[1., 0.], [0., 1.]])
        sx_mat = np.array([[0., 1.], [1., 0.]])
        sy_mat = np.array([[0., -1.j], [1.j, 0.]])
        sz_mat = np.array([[1., 0.], [0., -1.]])
        self.assertTrue(np.array_equal(np.round(self.si.calculate_matrix(), 7), si_mat))
        self.assertTrue(np.array_equal(np.round(self.sx.calculate_matrix(), 7), sx_mat))
        self.assertTrue(np.array_equal(np.round(self.sy.calculate_matrix(), 7), sy_mat))
        self.assertTrue(np.array_equal(np.round(self.sz.calculate_matrix(), 7), sz_mat))

    def test_basis_normalization(self):
        """ Tests that normalization of unit vectors is correct for basis Observables """
        self.assertAlmostEqual(abs(self.si), 1.0)
        self.assertAlmostEqual(abs(self.sx), 1.0)
        self.assertAlmostEqual(abs(self.sy), 1.0)
        self.assertAlmostEqual(abs(self.sz), 1.0)

    def test_basis_orthogonality(self):
        """ Tests that the dot product correctly predicts orthonormal bases"""
        self.assertAlmostEqual(self.si * self.sx, 0.)
        self.assertAlmostEqual(self.si * self.sy, 0.)
        self.assertAlmostEqual(self.si * self.sz, 0.)
        self.assertAlmostEqual(self.sx * self.sy, 0.)
        self.assertAlmostEqual(self.sx * self.sz, 0.)
        self.assertAlmostEqual(self.sy * self.sz, 0.)


class TestCalculate_cartesian_vector_from_matrix(TestCase):
    def test_basis_conversion(self):
        i = np.array([[1, 0], [0, 1]])
        x = np.array([[0, 1], [1, 0]])
        y = 1j * np.array([[0, -1], [1, 0]])
        z = np.array([[1, 0], [0, -1]])
        self.assertTrue(np.array_equal(la.calculate_cartesian_spin_vector_from_matrix(i), (1, 0, 0, 0)))
        self.assertTrue(np.array_equal(la.calculate_cartesian_spin_vector_from_matrix(x), (0, 1, 0, 0)))
        self.assertTrue(np.array_equal(la.calculate_cartesian_spin_vector_from_matrix(y), (0, 0, 1, 0)))
        self.assertTrue(np.array_equal(la.calculate_cartesian_spin_vector_from_matrix(z), (0, 0, 0, 1)))


class TestCommutator(TestCase):
    def test_cartesian_commutator(self):
        # Define Spin Basis Operators
        s0 = la.CartesianSpinOperator((0, 0, 0, 0))
        si = la.CartesianSpinOperator((1, 0, 0, 0))
        sx = la.CartesianSpinOperator((0, 1, 0, 0))
        sy = la.CartesianSpinOperator((0, 0, 1, 0))
        sz = la.CartesianSpinOperator((0, 0, 0, 1))
        self.assertEqual(s0, la.commutator(si, sx))
        self.assertEqual(la.commutator(sx, sy), -1 * la.commutator(sy, sx))
        self.assertEqual(2j*sz, la.commutator(sx, sy))

    def test_spherical_commutator(self):
        # Define Spin Basis Operators
        s0 = la.SphericalSpinOperator((0, 0, 0, 0))
        si = la.SphericalSpinOperator((1, 0, 0, 0))
        sx = la.SphericalSpinOperator((0, 1, np.pi/2, 0))
        sy = la.SphericalSpinOperator((0, 1, np.pi/2, np.pi/2))
        sz = la.SphericalSpinOperator((0, 1, 0, 0))
        self.assertEqual(s0, la.commutator(si, sx))
        self.assertEqual(la.commutator(sx, sy), -1 * la.commutator(sy, sx))
        self.assertEqual(2j * sz, la.commutator(sx, sy))

    def test_mixed_commutator(self):
        # Define Spin Basis Operators
        s0c = la.CartesianSpinOperator((0, 0, 0, 0))
        sic = la.CartesianSpinOperator((1, 0, 0, 0))
        sxc = la.CartesianSpinOperator((0, 1, 0, 0))
        syc = la.CartesianSpinOperator((0, 0, 1, 0))
        szc = la.CartesianSpinOperator((0, 0, 0, 1))
        # Define Spin Basis Operators
        s0s = la.SphericalSpinOperator((0, 0, 0, 0))
        sis = la.SphericalSpinOperator((1, 0, 0, 0))
        sxs = la.SphericalSpinOperator((0, 1, np.pi / 2, 0))
        sys = la.SphericalSpinOperator((0, 1, np.pi / 2, np.pi / 2))
        szs = la.SphericalSpinOperator((0, 1, 0, 0))
        self.assertEqual(s0c, la.commutator(sis, sxc))
        self.assertEqual(la.commutator(sxs, syc), -1 * la.commutator(sys, sxc))
        self.assertEqual(2j * szs, la.commutator(sxc, sys))


if __name__ == '__main__':
    unittest.main()