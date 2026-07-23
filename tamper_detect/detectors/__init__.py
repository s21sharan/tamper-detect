"""Detector package — protocol, registry, and concrete detectors."""

from tamper_detect.detectors.base import Detector, DetectorContext, register, registry

__all__ = ["Detector", "DetectorContext", "register", "registry"]
