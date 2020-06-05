
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ESIC_TEST_DEPENDS
        FileCheck count not
        esic
        )

add_lit_testsuite(check-esic "Running the standalone-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${ESIC_TEST_DEPENDS}
        )
set_target_properties(check-esic PROPERTIES FOLDER "Tests")

add_lit_testsuites(ESIC ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ESIC_TEST_DEPENDS})
