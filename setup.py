import os

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithStartupHook(build_py):
    def run(self):
        super().run()
        self.copy_file("pytorch_inductor_sqlite_patch.pth", self.build_lib)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            os.path.join(self.build_lib, "pytorch_inductor_sqlite_patch.pth")
        ]


setup(cmdclass={"build_py": BuildWithStartupHook})
