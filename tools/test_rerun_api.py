#!/usr/bin/env python3
"""
Simple test script to check Rerun API availability.
"""

try:
    import rerun as rr
    print("Rerun imported successfully")
    
    # Test basic initialization
    rr.init("test", spawn=False)
    print("Rerun init successful")
    
    # Test coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    print("ViewCoordinates successful")
    
    # Test transform
    import numpy as np
    translation = np.array([1.0, 2.0, 3.0])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    # Try different ways to specify rotation
    try:
        rr.log("test1", rr.Transform3D(translation=translation, rotation=quaternion))
        print("Transform3D with quaternion array successful")
    except Exception as e:
        print(f"Transform3D with quaternion array failed: {e}")
    
    try:
        rr.log("test2", rr.Transform3D(translation=translation))
        print("Transform3D without rotation successful")
    except Exception as e:
        print(f"Transform3D without rotation failed: {e}")
    
    # Test points
    try:
        points = np.array([[0, 0, 0], [1, 1, 1]])
        rr.log("test3", rr.Points3D(positions=points))
        print("Points3D successful")
    except Exception as e:
        print(f"Points3D failed: {e}")
    
    # Test time
    try:
        rr.log("test4", rr.Points3D(positions=points), rr.Time.from_time(0))
        print("Time.from_time successful")
    except Exception as e:
        print(f"Time.from_time failed: {e}")
    
    print("All tests completed")
    
except ImportError as e:
    print(f"Failed to import rerun: {e}")
except Exception as e:
    print(f"Error: {e}") 