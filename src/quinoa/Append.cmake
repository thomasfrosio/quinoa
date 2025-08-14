set(QUINOA_HEADERS
    src/quinoa/Alignment.hpp
    src/quinoa/CommonArea.hpp
    src/quinoa/ExcludeViews.hpp
    src/quinoa/GridSearch.hpp
    src/quinoa/Logger.hpp
    src/quinoa/Metadata.hpp
    src/quinoa/Optimizer.hpp
    src/quinoa/Settings.hpp
    src/quinoa/PairwiseShift.hpp
    src/quinoa/PairwiseTilt.hpp
    src/quinoa/Reconstruction.hpp
    src/quinoa/RotationOffset.hpp
    src/quinoa/Stack.hpp
    src/quinoa/Types.hpp
    src/quinoa/Thickness.hpp
    src/quinoa/Utilities.hpp

    src/quinoa/SplineCurve.hpp
    src/quinoa/SplineGrid.hpp

    src/quinoa/CTF.hpp
    src/quinoa/CTFBaseline.hpp
    src/quinoa/CTFGrid.hpp
    src/quinoa/CTFPatches.hpp
    src/quinoa/CTFSimulate.hpp

#    src/quinoa/Tests.hpp
    src/quinoa/FiguresBCI.hpp
)

set(QUINOA_SOURCES_CXX
    src/quinoa/Logger.cpp
    src/quinoa/Metadata.cpp
    src/quinoa/Settings.cpp
    src/quinoa/Utilities.cpp

)

set(QUINOA_SOURCES_UNIFIED
    src/quinoa/Entry.cpp
#    src/quinoa/Utilities.cpp
    src/quinoa/Plot.cpp
    src/quinoa/Stack.cpp

    src/quinoa/ExcludeViews.cpp

    src/quinoa/Alignment.cpp
    src/quinoa/PairwiseShift.cpp
    src/quinoa/PairwiseTilt.cpp
    src/quinoa/RotationOffset.cpp
    src/quinoa/CTFBaseline.cpp
    src/quinoa/CTFCoarse.cpp
    src/quinoa/CTFPatches.cpp
    src/quinoa/CTFRefine.cpp
    src/quinoa/Thickness.cpp
#    src/quinoa/ProjectionMatching.cpp
    src/quinoa/Reconstruction.cpp
    src/quinoa/Test.cpp
)
