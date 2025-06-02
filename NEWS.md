# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] unreleased

* Initial release. Contains:
  * `discrete_kalman_filter_manifold` for constructing Kalman filters whose state is represented by `KalmanState` with basic `predict!` and `update!` operations defined on it.
  * A set of propagation and update rules: `EKFPropagator`, `EKFUpdater`, `UnscentedPropagator` and `UnscentedUpdater`.
  * Basic covariance estimation algorithms: `CovarianceMatchingMeasurementCovarianceAdapter`, `CovarianceMatchingProcessCovarianceAdapter`.
  * Two examples (`gen_car_data`, `gen_car_sphere_data`).
