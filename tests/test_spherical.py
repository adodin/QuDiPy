from unittest import TestCase
import unittest
import QuDiPy.coordinates.spherical as sp
import numpy as np
from numpy import pi, sin

x_s = (1., pi/2, 0)
y_s = (1., pi/2, pi/2)
z_s = (1., 0., 0.)
minus_x_s = (1., pi/2, pi)

x_c = (1., 0., 0.)
y_c = (0., 1., 0.)
z_c = (0., 0., 1.)
minus_x_c = (-1., 0., 0.)


class TestSphericalToCartesian(TestCase):
    def test_spherical_to_cartesian_single_3D_points(self):
        x_convert = sp.spherical_to_cartesian(x_s)
        y_convert = sp.spherical_to_cartesian(y_s)
        z_convert = sp.spherical_to_cartesian(z_s)
        minus_x_convert = sp.spherical_to_cartesian(minus_x_s)

        # Test Float Equality to 7 Decimal Places for each Component
        for xc, xe in zip(x_convert, x_c):
            self.assertAlmostEqual(xc, xe)
        for yc, ye in zip(y_convert, y_c):
            self.assertAlmostEqual(yc, ye)
        for zc, ze in zip(z_convert, z_c):
            self.assertAlmostEqual(zc, ze)
        for xc, xe in zip(minus_x_convert, minus_x_c):
            self.assertAlmostEqual(xc, xe)

    def test_spherical_to_cartesian_one_3D_complex_point(self):
        x_iy_spherical = (1. + 1.j, pi/2 + 1.j * pi/2, 1.j * pi/2)
        x_iy_convert = sp.spherical_to_cartesian(x_iy_spherical)
        x_iy_cart = (1., 1.j, 0.)
        self.assertTrue(np.array_equal(np.round(x_iy_convert, 7), np.round(x_iy_cart, 7)))

    def test_spherical_to_cartesian_one_4D_point(self):
        x_spherical = (1., 1., pi / 2, 0.)
        y_spherical = (1., 1., pi / 2, pi / 2)
        z_spherical = (1., 1., 0., 0.)
        x_convert = sp.spherical_to_cartesian(x_spherical)
        y_convert = sp.spherical_to_cartesian(y_spherical)
        z_convert = sp.spherical_to_cartesian(z_spherical)
        x_cart = (1., 1., 0., 0.)
        y_cart = (1., 0., 1., 0.)
        z_cart = (1., 0., 0., 1.)

        for xconv, xcart in zip(x_convert, x_cart):
            self.assertAlmostEqual(xconv, xcart)

        for yconv, ycart in zip(y_convert, y_cart):
            self.assertAlmostEqual(yconv, ycart)

        for zconv, zcart in zip(z_convert, z_cart):
            self.assertAlmostEqual(zconv, zcart)

    def test_spherical_to_cartesian_one_4D_complex_point(self):
        x_iy_spherical = (1. + 1.j, 1. + 1.j, pi / 2 + 1.j * pi / 2, 1.j * pi / 2)
        x_iy_convert = sp.spherical_to_cartesian(x_iy_spherical)
        x_iy_cart = (1. + 1.j, 1., 1.j, 0.)
        self.assertTrue(np.array_equal(np.round(x_iy_convert, 7), np.round(x_iy_cart, 7)))

    def test_spherical_to_cartesian_3D_array(self):
        r_array = (1., 0.5, 0.0)
        theta_array = (0.0, pi/2)
        phi_array = (0.0, pi/2)
        spherical_grid = np.meshgrid(r_array, theta_array, phi_array, indexing='ij')
        x_conv, y_conv, z_conv = sp.spherical_to_cartesian(spherical_grid)
        x_grid = np.array([[[0., 0.], [1., 0.]], [[0., 0.], [0.5, 0.]], [[0., 0.], [0., 0.]]])
        y_grid = np.array([[[0., 0.], [0., 1.]], [[0., 0.], [0., 0.5]], [[0., 0.], [0., 0.]]])
        z_grid = np.array([[[1., 1.], [0., 0.]], [[0.5, 0.5], [0., 0.]], [[0., 0.], [0., 0.]]])
        self.assertTrue(np.array_equal(np.round(x_conv, 7), x_grid))
        self.assertTrue(np.array_equal(np.round(y_conv, 7), y_grid))
        self.assertTrue(np.array_equal(np.round(z_conv, 7), z_grid))

    def test_spherical_to_cartesian_3D_complex_array(self):
        r_array = (1. + 1.j, 0.5, 0.0)
        theta_array = (0.0, pi / 2 + 1.j * pi /2)
        phi_array = (0.0, pi / 2 + 1.j * pi / 2)
        spherical_grid = np.meshgrid(r_array, theta_array, phi_array, indexing='ij')
        x_conv, y_conv, z_conv = sp.spherical_to_cartesian(spherical_grid)
        x_grid = np.array([[[0., 0.], [1. + 1.j, 0.]], [[0., 0.], [0.5, 0.]], [[0., 0.], [0., 0.]]])
        y_grid = np.array([[[0., 0.], [0., 1. + 1.j]], [[0., 0.], [0., 0.5]], [[0., 0.], [0., 0.]]])
        z_grid = np.array([[[1. + 1.j, 1. +1.j], [0., 0.]], [[0.5, 0.5], [0., 0.]], [[0., 0.], [0., 0.]]])
        self.assertTrue(np.array_equal(np.round(x_conv, 7), x_grid))
        self.assertTrue(np.array_equal(np.round(y_conv, 7), y_grid))
        self.assertTrue(np.array_equal(np.round(z_conv, 7), z_grid))


class TestCartesianToSpherical(TestCase):
    def test_cartesian_to_spherical_one_3D_point(self):
        x_cart = (1., 0., 0.)
        x_m_cart = (-1., 0., 0.)
        y_cart = (0., 1., 0.)
        z_cart = (0., 0., 1.)
        x_conv = sp.cartesian_to_spherical(x_cart)
        x_m_conv = sp.cartesian_to_spherical(x_m_cart)
        y_conv = sp.cartesian_to_spherical(y_cart)
        z_conv = sp.cartesian_to_spherical(z_cart)
        x_spherical = (1., pi / 2, 0.)
        x_m_spherical = (1., pi / 2, pi)
        y_spherical = (1., pi / 2, pi / 2)
        z_spherical = (1., 0., 0.)
        self.assertTrue(np.array_equal(np.round(x_conv, 7), np.round(x_spherical, 7)))
        self.assertTrue(np.array_equal(np.round(x_m_conv, 7), np.round(x_m_spherical, 7)))
        self.assertTrue(np.array_equal(np.round(y_conv, 7), np.round(y_spherical, 7)))
        self.assertTrue(np.array_equal(np.round(z_conv, 7), np.round(z_spherical, 7)))

    def test_cartesian_to_spherical_one_3D_complex_point(self):
        x_iy_cart = (1., 1.j, 0.)
        x_iy_conv = sp.cartesian_to_spherical(x_iy_cart)
        x_iy_spherical = (1. + 1.j, pi / 2 + 1.j * pi / 2, 1.j * pi / 2 )
        self.assertTrue(np.array_equal(np.round(x_iy_conv, 7), np.round(x_iy_spherical, 7)))

    def test_cartesian_to_spherical_one_4D_point(self):
        x_cart = (1., 1., 0., 0.)
        y_cart = (1., 0., 1., 0.)
        z_cart = (0., 0., 0., 1.)
        x_conv = sp.cartesian_to_spherical(x_cart)
        y_conv = sp.cartesian_to_spherical(y_cart)
        z_conv = sp.cartesian_to_spherical(z_cart)
        x_spherical = (1., 1., pi / 2, 0.)
        y_spherical = (1., 1., pi / 2, pi / 2)
        z_spherical = (0., 1., 0., 0.)
        self.assertTrue(np.array_equal(np.round(x_conv, 7), np.round(x_spherical, 7)))
        self.assertTrue(np.array_equal(np.round(y_conv, 7), np.round(y_spherical, 7)))
        self.assertTrue(np.array_equal(np.round(z_conv, 7), np.round(z_spherical, 7)))

    def test_cartesian_to_spherical_one_4D_complex_point(self):
        x_iy_cart = (1. + 1.j, 1., 1.j, 0.)
        x_iy_conv = sp.cartesian_to_spherical(x_iy_cart)
        x_iy_spherical = (1. + 1.j, 1. + 1.j, pi / 2 + 1.j * pi / 2, 1.j * pi / 2)
        self.assertTrue(np.array_equal(np.round(x_iy_conv, 7), np.round(x_iy_spherical, 7)))


'''class TestSphericalToCartesianMesh(TestCase):
    def test_spherical_to_cartesian_mesh(self):
        r_array = (1., 0.5, 0.0)
        theta_array = (0.0, pi / 2)
        phi_array = (0.0, pi / 2)
        x_conv, y_conv, z_conv = sp.spherical_to_cartesian_mesh((r_array, theta_array, phi_array))
        x_grid = np.array([[[0., 0.], [1., 0.]], [[0., 0.], [0.5, 0.]], [[0., 0.], [0., 0.]]])
        y_grid = np.array([[[0., 0.], [0., 1.]], [[0., 0.], [0., 0.5]], [[0., 0.], [0., 0.]]])
        z_grid = np.array([[[1., 1.], [0., 0.]], [[0.5, 0.5], [0., 0.]], [[0., 0.], [0., 0.]]])
        self.assertTrue(np.array_equal(np.round(x_conv, 7), x_grid))
        self.assertTrue(np.array_equal(np.round(y_conv, 7), y_grid))
        self.assertTrue(np.array_equal(np.round(z_conv, 7), z_grid))'''


class TestCalculateSphericalVolumeElement(TestCase):
    def test_one_spherical_volume_element(self):
        r = 1.
        r0 = 0.
        theta = pi/2
        theta0 = 0
        diffs = (1., 1., 1.)
        self.assertAlmostEqual(sp.calculate_spherical_volume_element(r, theta, diffs), 1.)
        self.assertAlmostEqual(sp.calculate_spherical_volume_element(r0, theta, diffs), 0.)
        self.assertAlmostEqual(sp.calculate_spherical_volume_element(r, theta0, diffs), 0.)

    def test_grid_spherical_volume_element(self):
        r = (0., 1.)
        theta = (0., pi/2)
        r, theta = np.meshgrid(r, theta, indexing='ij')
        diffs = (np.ones_like(r), np.ones_like(r), np.ones_like(r))
        volume = np.array([[0., 0.], [0., 1.]])
        calc_volume = sp.calculate_spherical_volume_element(r, theta, diffs)
        self.assertTrue(np.array_equal(np.round(calc_volume, 7), volume))


class TestCalculateSphericalGradient(TestCase):
    def test_gradient_radial(self):
        r_array = np.linspace(0., 1., 20)
        theta_array = np.linspace(0., pi, 10)
        phi_array = np.linspace(0., 2*pi, 10)
        r_grid, t_grid, p_grid = np.meshgrid(r_array, theta_array, phi_array, indexing='ij')
        funct = r_grid
        correct_gradient = (np.ones_like(r_grid), np.zeros_like(t_grid), np.zeros_like(p_grid))
        calculated_gradient = sp.calculate_spherical_gradient(funct, (r_array, theta_array, phi_array),
                                                              (r_grid, t_grid, p_grid))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))

    def test_gradient_radial_quadratic(self):
        r_array = np.linspace(0., 1., 20)
        theta_array = np.linspace(0., pi, 10)
        phi_array = np.linspace(0., 2 * pi, 10)
        r_grid, t_grid, p_grid = np.meshgrid(r_array, theta_array, phi_array, indexing='ij')
        funct = r_grid ** 2
        correct_gradient = (2 * r_grid, np.zeros_like(t_grid), np.zeros_like(p_grid))
        calculated_gradient = sp.calculate_spherical_gradient(funct, (r_array, theta_array, phi_array),
                                                              (r_grid, t_grid, p_grid))
        for correct, calculated in zip(correct_gradient, calculated_gradient):
            self.assertTrue(np.array_equal(np.round(calculated, 7), np.round(correct, 7)))


class TestCalculateSphericalDivergence(TestCase):
    def test_divergence_radial(self):
        r_array = np.linspace(1E-12, 1., 20)
        theta_array = np.linspace(0., pi, 10)
        phi_array = np.linspace(0., 2*pi, 10)
        r_grid, t_grid, p_grid = np.meshgrid(r_array, theta_array, phi_array, indexing='ij')
        vector_funct = (r_grid, np.zeros_like(t_grid), np.zeros_like(p_grid))
        correct_divergence = 3 * np.ones_like(r_grid)
        calculated_divergence = sp.calculate_spherical_divergence(vector_funct, (r_array, theta_array, phi_array),
                                                                  (r_grid, t_grid, p_grid))
        self.assertTrue(np.array_equal(np.round(calculated_divergence, 7), correct_divergence))

    def test_divergence_azimuthal(self):
        r_array = np.linspace(1E-12, 1., 20)
        theta_array = np.linspace(1E-12, pi, 10)
        phi_array = np.linspace(0., 2 * pi, 10)
        r_grid, t_grid, p_grid = np.meshgrid(r_array, theta_array, phi_array, indexing='ij')
        vector_funct = (np.zeros_like(r_grid), np.zeros_like(t_grid), r_grid * sin(t_grid))
        correct_divergence = np.zeros_like(r_grid)
        calculated_divergence = sp.calculate_spherical_divergence(vector_funct, (r_array, theta_array, phi_array),
                                                                  (r_grid, t_grid, p_grid))
        self.assertTrue(np.array_equal(np.round(calculated_divergence, 7), correct_divergence))


if __name__ == '__main__':
    unittest.main()
