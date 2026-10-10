"""Narrow compatibility import for the vendored BODex cuRobo checkout.

The legacy code imports two RSS helpers although the current AutoDex
environment no longer installs rsslib. This demo-only package supplies those
imports without editing the shared grasp-generation implementation.
"""
