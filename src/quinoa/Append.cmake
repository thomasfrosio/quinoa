set(QUINOA_HEADERS
    src/quinoa/Alignment.hpp
    src/quinoa/CommonArea.hpp
    src/quinoa/ExcludeViews.hpp
    src/quinoa/GridSearch.hpp
    src/quinoa/Logger.hpp
    src/quinoa/Metadata.hpp
    src/quinoa/Optimizer.hpp
    src/quinoa/Options.hpp
    src/quinoa/PairwiseShift.hpp
    src/quinoa/PairwiseTilt.hpp
    src/quinoa/Reconstruction.hpp
    src/quinoa/RotationOffset.hpp
    src/quinoa/Stack.hpp
    src/quinoa/Types.hpp
    src/quinoa/Thickness.hpp
    src/quinoa/Utilities.hpp
    src/quinoa/YAML.hpp

    src/quinoa/CubicGrid.hpp
    src/quinoa/CTF.hpp
#    src/quinoa/Tests.hpp
)

set(QUINOA_SOURCES_CXX
    src/quinoa/Logger.cpp
    src/quinoa/Metadata.cpp
    src/quinoa/Options.cpp

)

set(QUINOA_SOURCES_UNIFIED
    src/quinoa/Alignment.cpp
    src/quinoa/Entry.cpp
    src/quinoa/PairwiseShift.cpp
    src/quinoa/PairwiseTilt.cpp
#    src/quinoa/ProjectionMatching.cpp
    src/quinoa/Reconstruction.cpp
    src/quinoa/RotationOffset.cpp
    src/quinoa/Stack.cpp
#    src/quinoa/Thickness.cpp

    src/quinoa/CTFPatches.cpp
    src/quinoa/CTFCoarse.cpp
#    src/quinoa/CTFAverage.cpp
#    src/quinoa/CTFGlobal.cpp
    src/quinoa/CTFBackground.cpp
)
