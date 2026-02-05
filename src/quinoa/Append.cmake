set(QUINOA_HEADERS
    src/quinoa/CommonFOV.hpp
    src/quinoa/ExcludeViews.hpp
    src/quinoa/GridSearch.hpp
    src/quinoa/Logger.hpp
    src/quinoa/Metadata.hpp
    src/quinoa/Optimizer.hpp
    src/quinoa/Reconstruct.hpp
    src/quinoa/Settings.hpp
    src/quinoa/SplineCurve.hpp
    src/quinoa/SplineGrid.hpp
    src/quinoa/Stack.hpp
    src/quinoa/Thickness.hpp
    src/quinoa/Types.hpp
    src/quinoa/Utilities.hpp

    src/quinoa/align/Align.hpp
    src/quinoa/align/Coarse.hpp
    src/quinoa/align/Projection.hpp
    src/quinoa/align/Rotation.hpp

    src/quinoa/ctf/CTF.hpp
    src/quinoa/ctf/Baseline.hpp
    src/quinoa/ctf/Grid.hpp
    src/quinoa/ctf/Patches.hpp
    src/quinoa/ctf/Simulate.hpp
    src/quinoa/ctf/Thickness.hpp

)

set(QUINOA_SOURCES_CXX
    src/quinoa/Logger.cpp
    src/quinoa/Metadata.cpp
    src/quinoa/Settings.cpp
    src/quinoa/Utilities.cpp
)

set(QUINOA_SOURCES_UNIFIED
    src/quinoa/Entry.cpp
    src/quinoa/ExcludeViews.cpp
    src/quinoa/Plot.cpp
    src/quinoa/Reconstruct.cpp
    src/quinoa/Stack.cpp
    src/quinoa/Thickness.cpp

    src/quinoa/align/Align.cpp
    src/quinoa/align/Coarse.cpp
    src/quinoa/align/Projection.cpp
    src/quinoa/align/Rotation.cpp

    src/quinoa/ctf/CTF.cpp
    src/quinoa/ctf/Baseline.cpp
    src/quinoa/ctf/Coarse.cpp
    src/quinoa/ctf/Patches.cpp
    src/quinoa/ctf/Refine.cpp

)
