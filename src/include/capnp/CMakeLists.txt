# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_custom_target(capnp-support ALL DEPENDS EsiCoreAnnotations.capnp)
configure_file(EsiCoreAnnotations.capnp ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/EsiCoreAnnotations.capnp COPYONLY)
