class HinFlowReference:
    name = "hinflow"
    kind = "online_learning_reference"
    leaderboard = "none"
    selects = "not_a_selector"
    paper_method = False
    allowed_information = ("hindsight_relabel", "flow_planner", "original_backbone")


REFERENCE = HinFlowReference()
