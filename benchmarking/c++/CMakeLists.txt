# Copyright (C) 2011-2013  Istituto Italiano di Tecnologia, Massachussets Institute of Techology
# Authors: Elena Ceseracciu <elena.ceseracciu@iit.it>, Matteo Santoro <msantoro@mit.edu>
include_directories(${Gurls++_INCLUDE_DIRS})

add_executable(primal primal.cpp)
target_link_libraries(primal ${Gurls++_LIBRARIES})

add_executable(dual dual.cpp)
target_link_libraries(dual ${Gurls++_LIBRARIES})

add_executable(gaussian gaussian.cpp)
target_link_libraries(gaussian ${Gurls++_LIBRARIES})

