from xarm.wrapper import XArmAPI

class RoboticArm:
    def __init__(self, ip='192.168.1.111'):
        self.api = XArmAPI(ip)
        self.zLowered=137
        self.zRaised=152
        self.roll=0
        self.pitch = 200
        self.yaw = 0
        self.speed=100
        
        self.min_x = 150  # 100 + 50
        self.max_x = 323  # 373 - 50
        
    def set_position(self, x, y, draw):
        if draw:
            # Calculate the midpoint of the x-axis drawing region.
            mid_x = (self.min_x + self.max_x) / 2
            
            # If the arm is drawing in the top half of the x-axis, lower the z height by 1.
            if x >= mid_x:
                z = self.zLowered - 1
            else:
                z = self.zLowered
            
            self.api.set_position(x, y, z, self.roll, self.pitch, self.yaw, speed=self.speed)
        else:
            self.api.set_position(x, y, self.zRaised, self.roll, self.pitch, self.yaw, speed=self.speed)