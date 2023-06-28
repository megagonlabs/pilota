#!/usr/bin/env python3

import unittest

import pilota.cli


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_cli(self):
        pilota.cli.get_opts()


if __name__ == "__main__":
    unittest.main()
