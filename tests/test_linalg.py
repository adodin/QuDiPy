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
        self.si = la.CartesianSpinObservable(self.i)
        self.si2 = la.CartesianSpinObservable(self.i)
        self.sx = la.CartesianSpinObservable(self.x)
        self.sy = la.CartesianSpinObservable(self.y)
        self.sz = la.CartesianSpinObservable(self.z)
        self.basis_operators = [self.si, self.sx, self.sy, self.sz]
        self.srands = [la.CartesianSpinObservable(rand) for rand in self.rands]
        self.operators = self.basis_operators + self.srands
        pass

    def test_init(self):
        """ Tests that all constructors initialize the correct Observables"""
        for op, vec in zip(self.basis_operators, self.vectors):
            self.assertEqual(op.vector, vec)

    def test_init_length_assertion(self):
        """ Tests that the constructor correctly raises initialization error if the wrong length input provided """
        with self.assertRaises(AssertionError):
            la.CartesianSpinObservable((1., 0., 0.))
        with self.assertRaises(AssertionError):
            la.CartesianSpinObservable((1., 0., 0., 0., 0., 0.))

    def test_init_real_assertion(self):
        """ Tests that the constructor correctly raises initialization error if complex inputs provided """
        with self.assertRaises(AssertionError):
            la.CartesianSpinObservable((1.j, 0., 0., 0.))
        with self.assertRaises(AssertionError):
            la.CartesianSpinObservable((0., 1. + 1.j, 0., 0.))

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
        self.si = la.SphericalSpinObservable(self.i)
        self.sx = la.SphericalSpinObservable(self.x)
        self.sy = la.SphericalSpinObservable(self.y)
        self.sz = la.SphericalSpinObservable(self.z)
        pass

    def test_init(self):
        """ Tests the initialization of """
        self.assertEqual(self.si.vector, self.i)
        self.assertEqual(self.sx.vector, self.x)
        self.assertEqual(self.sy.vector, self.y)
        self.assertEqual(self.sz.vector, self.z)

    def test_init_length_assertion(self):
        """ Tests that the constructor correctly raises initialization error if the wrong length input provided """
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., 0., 0.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., 0., 0., 0., 0., 0.))

    def test_init_real_assertion(self):
        """ Tests that the constructor correctly raises initialization error if complex inputs provided """
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1.j, 0., 0., 0.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((0., 1. + 1.j, 0., 0.))

    def test_init_range_assertion(self):
        """ Tests that the initializer catches invalid spherical coordinate inputs"""
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., -1., 0., 0.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., 0., -1., 0.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., 0., 10., 0.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., 0., 0., -1.))
        with self.assertRaises(AssertionError):
            la.SphericalSpinObservable((1., 0., 0., 10.))

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


if __name__ == '__main__':
    unittest.main()