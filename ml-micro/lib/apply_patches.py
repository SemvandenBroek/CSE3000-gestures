import os
import platform

env = DefaultEnvironment()

err_redir = "/dev/null" if platform.system() == "Linux" else "nul"

# Because of a stupid bug in the flatbuffers library and a version mismatch with TFLite-Micro when flatbuffers is updated, we have to apply this patch
def apply_git_patches():
    os.chdir("../lib/flatbuffers/")

    applied = os.system(f"git apply ../flatbuffers.patch 2>{err_redir}")
    if applied == 0:
        print("Applied flatbuffers patch")
    else:
        print("Flatbuffers patch is already applied!")

apply_git_patches()

