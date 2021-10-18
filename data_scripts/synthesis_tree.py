from collections import deque

class ReactionNode:
    '''
    represents a reaction in the SynthesisTree
    similar to a binary tree node

    by default, self.left will be filled first (uni-molecular reaction)
    and self.right if bi-molecular
    '''
    def __init__(
        self, template_str
    ):
        self.template_str = template_str
        # can also store template_idx
        self.left = None
        self.right = None
        self.head = None

class MoleculeNode:
    '''
    represents a molecule in the SythesisTree
    similar to a LinkedList node
    '''
    def __init__(
        self, smi
    ):
        self.smi = smi
        self.prev = None
        self.next = None

class SynthesisTree:
    def __init__(self):
        self.root = None # store the final product MoleculeNode
        self.actions = [] # "EXPAND", "ADD", "MERGE", "STOP"
        self.molecules = [] # MoleculeNode (both reactant & product) in order of addition
        self.reactions = [] # ReactionNode, in order of addition
        self.state = deque(maxlen=2) # at most two molecules, append from left, pop from right

    def eval_state(self):
        # 0-th index element is more recently added to tree

        if len(self.state) == 0:
            return []

        elif len(self.state) == 1:
            return [self.state[0].smi]

        else:
            return [self.state[0].smi, self.state[1].smi]

    def execute_action(self, action, template_str, rct1_smi, rct2_smi, prod_smi):
        # assume the action is valid (?) or may need try except catch
        # 0 --> "ADD", 1 --> "EXPAND", 2 --> "MERGE", 3 --> "END"

        if action == 3: # END
            self.root = self.molecules[-1]

        else:
            prod_node = MoleculeNode(prod_smi)
            rxn_node = ReactionNode(template_str)
            rxn_node.head = prod_node
            prod_node.prev = rxn_node

            if action == 0: # ADD
                rct1_node = MoleculeNode(rct1_smi)
                rct1_node.next = rxn_node
                self.molecules.append(rct1_node)
                rxn_node.left = self.molecules[-1]

                if rct2_smi:
                    rct2_node = MoleculeNode(rct2_smi)
                    rct2_node.next = rxn_node
                    self.molecules.append(rct2_node)
                    rxn_node.right = rct2_node

                self.state.appendleft(prod_node)
                # new prod_node becomes most_recent node,
                # and future expand can only act on this prod_node, until merge is sampled
                # the previous prod_node (hanging sub-tree root) can never be expanded again (only merged)

            elif action == 1: # EXPAND
                rct1_node = self.molecules[-1]
                rct1_node.next = rxn_node
                rxn_node.left = rct1_node

                if rct2_smi:
                    rct2_node = MoleculeNode(rct2_smi)
                    rct2_node.next = rxn_node
                    self.molecules.append(rct2_node)
                    rxn_node.right = rct2_node

                # replace most recently added molecule with new product
                self.state.popleft()
                self.state.appendleft(prod_node)

            elif action == 2: # MERGE
                rct1_node = self.molecules[-1]
                rct1_node.next = rxn_node
                rxn_node.left = rct1_node

                rct2_node = self.state[-1]
                rct2_node.next = rxn_node
                rxn_node.right = rct2_node

                self.state.appendleft(prod_node)
                # if merge, state now only should have 1 molecule (the new product)
                # pop the previous molecule node
                self.state.pop()

            else:
                raise ValueError(f'invalid action: {action}')

            self.reactions.append(rxn_node)
            self.molecules.append(prod_node)

        self.actions.append(action)