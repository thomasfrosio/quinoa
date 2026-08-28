set(QUINOA_HEADERS
    src/quinoa/GridSearch.hpp
    src/quinoa/Logger.hpp
    src/quinoa/Metadata.hpp
    src/quinoa/Optimizer.hpp
    src/quinoa/Settings.hpp
    src/quinoa/SplineCurve.hpp
    src/quinoa/SplineGrid.hpp
    src/quinoa/Stack.hpp
    src/quinoa/Types.hpp
    src/quinoa/Utilities.hpp

    src/quinoa/preprocessing/ExcludeViews.hpp
    src/quinoa/preprocessing/Run.hpp

    src/quinoa/ctf/CTF.hpp
    src/quinoa/ctf/Baseline.hpp
    src/quinoa/ctf/Grid.hpp
    src/quinoa/ctf/Patches.hpp
    src/quinoa/ctf/Refine.hpp
    src/quinoa/ctf/Run.hpp
)

set(QUINOA_SOURCES_CXX
    src/quinoa/Logger.cpp
    src/quinoa/Metadata.cpp
    src/quinoa/Settings.cpp
    src/quinoa/Utilities.cpp
)

set(QUINOA_SOURCES_UNIFIED
    src/quinoa/Main.cpp
    src/quinoa/Plot.cpp
    src/quinoa/Stack.cpp

    src/quinoa/preprocessing/ExcludeViews.cpp
    src/quinoa/preprocessing/Run.cpp

    src/quinoa/ctf/Baseline.cpp
    src/quinoa/ctf/Coarse.cpp
    src/quinoa/ctf/Patches.cpp
    src/quinoa/ctf/Refine.cpp
    src/quinoa/ctf/Run.cpp
)

if(NOT QN_CTF_ONLY)
    list(APPEND QUINOA_HEADERS
        src/quinoa/align/Run.hpp
        src/quinoa/align/Tilter.hpp
        src/quinoa/align/CommonFOV.hpp
        src/quinoa/align/Projection.hpp
        src/quinoa/align/Thickness.hpp

        src/quinoa/postprocessing/BackwardProjection.hpp
        src/quinoa/postprocessing/FilterStack.hpp
        src/quinoa/postprocessing/FourierInsertion.hpp
        src/quinoa/postprocessing/Run.hpp
        src/quinoa/postprocessing/Utilities.hpp
    )

    list(APPEND QUINOA_SOURCES_UNIFIED
        src/quinoa/align/Run.cpp
        src/quinoa/align/Tilter.cpp
        src/quinoa/align/Projection.cpp
        src/quinoa/align/Thickness.cpp

        src/quinoa/postprocessing/FilterStack.cpp
        src/quinoa/postprocessing/Run.cpp
    )
endif ()
