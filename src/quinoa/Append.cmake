set(QUINOA_SOURCES
    src/quinoa/EntryPoint.cpp

    src/quinoa/Exception.h
    src/quinoa/Exception.cpp
    src/quinoa/Types.h

    src/quinoa/io/Logging.h
    src/quinoa/io/Logger.cpp
    src/quinoa/io/Options.h
    src/quinoa/io/Options.cpp
    src/quinoa/io/YAML.h

    src/quinoa/core/Alignment.h
    src/quinoa/core/Alignment.cpp
    src/quinoa/core/Metadata.cpp
    src/quinoa/core/Metadata.h
    src/quinoa/core/Stack.hpp
    src/quinoa/core/Stack.cpp
    src/quinoa/core/Optimizer.hpp
    src/quinoa/core/Utilities.h
    src/quinoa/core/CommonArea.hpp

    src/quinoa/core/Ewise.hpp
    src/quinoa/core/Ewise.cu

    src/quinoa/core/Thickness.hpp
    src/quinoa/core/Thickness.cpp

    src/quinoa/core/CTF.hpp
    src/quinoa/core/CTFAverage.cpp
    src/quinoa/core/CTFGlobal.cpp

    src/quinoa/core/GridSearch1D.hpp
    src/quinoa/core/CubicGrid.hpp

    src/quinoa/core/PairwiseShift.hpp
    src/quinoa/core/PairwiseShift.cpp
    src/quinoa/core/GlobalRotation.hpp
    src/quinoa/core/GlobalRotation.cpp
#    src/quinoa/core/ProjectionMatching.h
    src/quinoa/core/ProjectionMatching.hpp
#    src/quinoa/core/ProjectionMatching.cpp
    src/quinoa/core/ProjectionMatching2.cpp

    )
