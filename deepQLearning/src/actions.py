class Actions:
    HOlD_KEY = 'holdKey'
    RELEASE_KEY = 'releaseKey'
    AMOUNT_OF_ACTIONS = 2

    def __init__(self):
        pass

    def getAnother(self, action):
        if action == Actions.HOlD_KEY:
            return Actions.RELEASE_KEY
        else:
            return Actions.HOlD_KEY
