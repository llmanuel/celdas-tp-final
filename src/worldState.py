from src.distance import Zones

class WorldState:
    def __init__(self, farAwayFormWall, distanceToGap, velocity, currentPositions, isDead = None):
        self.farAwayFormWall = farAwayFormWall
        self.distanceToGap = distanceToGap
        self.zone = Zones().getZoneAccordingToWalls(currentPositions)
        self.isDead = isDead
        self.velocity = velocity
        self.currentPositions = currentPositions

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, WorldState):
            return (
                (self.distanceToGap > 0) == (other.distanceToGap > 0) and
                self.isDead == other.isDead and
                self.zone == other.zone and
                self.similarVelocity(other)
            )
        return False

    def similarVelocity(self, other):
        def first(vel):
            return vel >= 6
        def second(vel):
            return 6 > vel >= 2
        def third(vel):
            return 2 > vel >= -2
        def fourth(vel):
            return -2 > vel >= -6
        def fifth(vel):
            return -6 > vel

        return (
            first(self.velocity) == first(other.velocity) or
            second(self.velocity) == second(other.velocity) or 
            third(self.velocity) == third(other.velocity) or 
            fourth(self.velocity) == fourth(other.velocity) or 
            fifth(self.velocity) == fifth(other.velocity)
        )
