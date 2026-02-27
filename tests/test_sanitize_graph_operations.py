import pytest

from agent.planner import PlannerAgent
from agent.planner_types import GraphOperation, GraphOperationCommand


@pytest.mark.parametrize(
    "ops, expected_len",
    [
        (
            [
                # duplicate ADD_NODE via node_data id
                GraphOperation(
                    command=GraphOperationCommand.ADD_NODE,
                    node_data={"id": "n1", "description": "a"},
                ),
                GraphOperation(
                    command=GraphOperationCommand.ADD_NODE,
                    node_data={"id": "n1", "description": "a-dup"},
                ),
                # invalid ADD_NODE (missing id)
                GraphOperation(command=GraphOperationCommand.ADD_NODE, node_data={"description": "x"}),
                # valid UPDATE_NODE
                GraphOperation(
                    command=GraphOperationCommand.UPDATE_NODE,
                    node_id="n1",
                    updates={"status": "completed"},
                ),
                # invalid UPDATE_NODE (missing updates)
                GraphOperation(command=GraphOperationCommand.UPDATE_NODE, node_id="n1", updates=None),
                # invalid REMOVE_NODE
                GraphOperation(command=GraphOperationCommand.REMOVE_NODE, node_id=None),
                # edge ops: duplicate ADD_EDGE
                GraphOperation(
                    command=GraphOperationCommand.ADD_EDGE,
                    edge_data={"from": "n1", "to": "n2", "type": "dep"},
                ),
                GraphOperation(
                    command=GraphOperationCommand.ADD_EDGE,
                    edge_data={"source": "n1", "target": "n2", "type": "dep"},
                ),
                # edge ops: invalid REMOVE_EDGE (missing endpoints)
                GraphOperation(command=GraphOperationCommand.REMOVE_EDGE, edge_data={"from": "n1"}),
                # edge ops: duplicate REMOVE_EDGE
                GraphOperation(
                    command=GraphOperationCommand.REMOVE_EDGE,
                    edge_data={"from": "n1", "to": "n2", "type": "dep"},
                ),
                GraphOperation(
                    command=GraphOperationCommand.REMOVE_EDGE,
                    edge_data={"from": "n1", "to": "n2", "type": "dep"},
                ),
            ],
            4,  # ADD_NODE(n1) + UPDATE_NODE + ADD_EDGE + REMOVE_EDGE
        ),
    ],
)
def test_sanitize_graph_operations_dedup_and_filter(ops, expected_len):
    # only need the instance method; __init__ isn't required for this helper
    agent = object.__new__(PlannerAgent)

    sanitized = agent._sanitize_graph_operations(ops)

    assert len(sanitized) == expected_len

    # ADD_NODE should have node_id backfilled from node_data['id']
    add_nodes = [op for op in sanitized if op.command == GraphOperationCommand.ADD_NODE]
    assert len(add_nodes) == 1
    assert add_nodes[0].node_id == "n1"

