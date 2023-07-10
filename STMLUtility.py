class STMLUtility:
    def __init_(self):
        pass

    def find_target_index(self, target: str, target_list: list):
        try:
            return target_list.index(target)
        except:
            return 0
