class Actions:
    HOlD_KEY = [1, 0]
    RELEASE_KEY = [0, 1]
    AMOUNT_OF_ACTIONS = 2

    def __init__(self):
        pass

    def getAnother(self, action):
        if action == Actions.HOlD_KEY:
            return Actions.RELEASE_KEY
        else:
            return Actions.HOlD_KEY
