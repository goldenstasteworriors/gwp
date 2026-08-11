"""Pure-CPU conformance codec for the SM-FET ``minimal-r1d`` schema.

This module implements the hardware-family *structural* alphabet frozen by
``sm_fet_minimal_r1d_schema_contract_20260812.md`` and its freeze addendum.
It deliberately does not create a metric codebook, a G1 capability profile,
runtime admission, or an executable controller profile.

Only Python's standard library is used.  Validation is fail-closed: callers
receive :class:`SchemaError` with a stable ``code`` rather than repaired,
clamped, modulo-reduced, or completion-filled data.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
import hashlib
import json
import math
import re
import struct
from typing import Any, Final


SCHEMA_ID: Final = "sm_fet_hard_grammar_v2_minimal_r1d_struct_v1"
WIRE_BYTES: Final = 32
CONSTRAINT_BYTES: Final = 15
HARD_TOKEN_SLOTS: Final = 5
COMPACT_BYTES: Final = 13

TRANSLATIONS: Final = 64 * 64 * 32
NONVERIFY_VARIANTS: Final = 1 + 512 + 512 * 32 + 512 * 32 * 4 * 8
VERIFY_VARIANTS: Final = 1 + 512 + 512 * 32
L_NONVERIFY: Final = TRANSLATIONS * NONVERIFY_VARIANTS
L_VERIFY: Final = TRANSLATIONS * VERIFY_VARIANTS
N: Final = 7 * 4 * L_NONVERIFY * 22
V: Final = 1 * 4 * L_VERIFY * 22
B_NONVERIFY: Final = 2 * N + 7 * N * N + N * (N - 1) // 2
B_VERIFY: Final = 3 * V + 3 * V * V + 2 * (V * (V - 1) // 2)
STRUCTURAL_STATES: Final = 1 + 11 * B_NONVERIFY + B_VERIFY
MAX_GLOBAL_RANK: Final = STRUCTURAL_STATES - 1
STRUCTURAL_LOG2: Final = "96.99142134161319"
MINIMUM_INDEX_BITS: Final = 97

EQUAL_BIT_MULTIPLIER: Final = 65_537
EQUAL_BIT_OFFSET: Final = 104_729
EQUAL_BIT_INVERSE: Final = 138_514_980_096_989_891_931_988_754_433

PUBLIC_STATE_MAGIC: Final = b"SMFETPES1"
PUBLIC_STATE_VERSION: Final = 1
PUBLIC_STATE_BYTES: Final = 226

SHARED_SEMANTIC_SCHEMA_ID: Final = "sm_fet_r1d_shared_semantic_key_v1"
TEXT_GOAL_SCHEMA_ID: Final = "sm_fet_r1d_text_goal_v1"
METRIC_MANIFEST_SCHEMA_ID: Final = "sm_fet_r1d_metric_codebook_manifest_v1"
AXIS_TABLE_UNIT_TOLERANCE: Final = float.fromhex("0x1.0000000000000p-40")

_IDENTIFIER_RE = re.compile(r"^[a-z0-9._/-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_HEX_RE = re.compile(r"^-?0x[0-9a-f]+\.[0-9a-f]+p[+-]?[0-9]+$")


class SchemaError(ValueError):
    """A deterministic, fail-closed schema validation failure."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


class Phase(IntEnum):
    APPROACH_ALIGN = 1
    ACQUIRE = 2
    CONTROLLED_CONTACT = 3
    REGRASP = 4
    RELEASE = 5
    VERIFY = 6


class Effect(IntEnum):
    NONE = 1
    TRANSLATE = 2
    ROTATE = 3
    ARTICULATE = 4
    FLOW = 5
    DEFORM = 6
    TRANSFER = 7


class ConstraintFunction(IntEnum):
    OPERATE = 1
    STABILIZE = 2
    MONITOR = 3


class Assignment(IntEnum):
    LEFT_RH56 = 1
    RIGHT_OMNIPICKER = 2
    MONITOR_ONLY = 3


class Role(IntEnum):
    NONE = 0
    SOURCE = 1
    TOOL = 2
    DESTINATION = 3
    SUPPORT = 4


class Part(IntEnum):
    PAD = 0
    N_A = 1
    BODY = 2
    HANDLE_OR_GRASP_FEATURE = 3
    LID_CAP = 4
    OPENING_OR_RIM = 5
    TOOL_TIP = 6
    ACTUATOR_OR_PLUNGER = 7
    WORKING_FACE_OR_PAD = 8


class LocatorKind(IntEnum):
    POINT = 1
    AXIS = 2
    FRAME = 3
    PATH = 4


class PathPrimitive(IntEnum):
    LINE = 1
    ARC_LEFT = 2
    ARC_RIGHT = 3
    RASTER_3 = 4


class Relation(IntEnum):
    N_A = 0
    SEPARATE = 1
    NEAR = 2
    TOUCHING = 3
    INSIDE = 4
    CONTAINS = 5
    ALIGNED_TO_RIM = 6
    SUPPORTED_BY = 7


class ConstraintType(str, Enum):
    LO = "LO"
    LS = "LS"
    RO = "RO"
    RS = "RS"
    ZM = "ZM"


class ClaimMode(str, Enum):
    EXCLUSIVE_TARGET = "EXCLUSIVE_TARGET"
    HOLD_SAME_TARGET = "HOLD_SAME_TARGET"
    READ_ONLY = "READ_ONLY"


class TargetNamespace(str, Enum):
    APERTURE_ACTUATION = "APERTURE_ACTUATION"
    WRIST_POSE = "WRIST_POSE"
    HELD_ENTITY = "HELD_ENTITY"
    STABILIZED_ENTITY = "STABILIZED_ENTITY"
    MONITORED_ENTITY = "MONITORED_ENTITY"
    SHARED_WBC = "SHARED_WBC"
    SELF_COLLISION = "SELF_COLLISION"


class TargetSelector(str, Enum):
    CONTROLLER_ENUM = "CONTROLLER_ENUM"
    LOCATOR_POINT = "LOCATOR_POINT"
    LOCATOR_AXIS = "LOCATOR_AXIS"
    LOCATOR_FULL = "LOCATOR_FULL"
    ENTITY_LOCATOR = "ENTITY_LOCATOR"
    CONSTANT = "CONSTANT"


HEADER_BLOCKS: Final[tuple[tuple[Phase, Effect], ...]] = (
    (Phase.APPROACH_ALIGN, Effect.NONE),
    (Phase.ACQUIRE, Effect.NONE),
    (Phase.CONTROLLED_CONTACT, Effect.NONE),
    (Phase.CONTROLLED_CONTACT, Effect.TRANSLATE),
    (Phase.CONTROLLED_CONTACT, Effect.ROTATE),
    (Phase.CONTROLLED_CONTACT, Effect.ARTICULATE),
    (Phase.CONTROLLED_CONTACT, Effect.FLOW),
    (Phase.CONTROLLED_CONTACT, Effect.DEFORM),
    (Phase.CONTROLLED_CONTACT, Effect.TRANSFER),
    (Phase.REGRASP, Effect.NONE),
    (Phase.RELEASE, Effect.NONE),
    (Phase.VERIFY, Effect.NONE),
)
_HEADER_TO_INDEX = {item: index for index, item in enumerate(HEADER_BLOCKS)}

NONVERIFY_TOPOLOGIES: Final[tuple[tuple[ConstraintType, ...], ...]] = (
    (ConstraintType.LO,),
    (ConstraintType.RO,),
    (ConstraintType.LO, ConstraintType.RO),
    (ConstraintType.LO, ConstraintType.RS),
    (ConstraintType.LS, ConstraintType.RO),
    (ConstraintType.LO, ConstraintType.ZM),
    (ConstraintType.RO, ConstraintType.ZM),
    (ConstraintType.LO, ConstraintType.LS),
    (ConstraintType.RO, ConstraintType.RS),
    (ConstraintType.RO, ConstraintType.RO),
)
VERIFY_TOPOLOGIES: Final[tuple[tuple[ConstraintType, ...], ...]] = (
    (ConstraintType.LS,),
    (ConstraintType.RS,),
    (ConstraintType.ZM,),
    (ConstraintType.LS, ConstraintType.RS),
    (ConstraintType.LS, ConstraintType.ZM),
    (ConstraintType.RS, ConstraintType.ZM),
    (ConstraintType.ZM, ConstraintType.ZM),
    (ConstraintType.RS, ConstraintType.RS),
)

_TYPE_FIELDS: Final = {
    ConstraintType.LO: (ConstraintFunction.OPERATE, Assignment.LEFT_RH56),
    ConstraintType.LS: (ConstraintFunction.STABILIZE, Assignment.LEFT_RH56),
    ConstraintType.RO: (ConstraintFunction.OPERATE, Assignment.RIGHT_OMNIPICKER),
    ConstraintType.RS: (ConstraintFunction.STABILIZE, Assignment.RIGHT_OMNIPICKER),
    ConstraintType.ZM: (ConstraintFunction.MONITOR, Assignment.MONITOR_ONLY),
}
_FIELDS_TYPE = {value: key for key, value in _TYPE_FIELDS.items()}


@dataclass(frozen=True, slots=True)
class Locator:
    kind: LocatorKind
    uv: int
    depth: int
    axis: int = 0
    roll: int = 0
    primitive: PathPrimitive | int = 0
    extent: int = 0


@dataclass(frozen=True, slots=True)
class Constraint:
    function: ConstraintFunction
    assignment: Assignment
    primary_role: Role
    part: Part
    locator: Locator
    secondary_role: Role = Role.NONE
    relation: Relation = Relation.N_A


@dataclass(frozen=True, slots=True)
class EventPacket:
    phase: Phase | None
    effect: Effect | None
    constraints: tuple[Constraint, ...] = ()

    @classmethod
    def null(cls, *, exhaustive_no_active: bool) -> "EventPacket":
        if type(exhaustive_no_active) is not bool or not exhaustive_no_active:
            raise SchemaError(
                "NULL_NOT_EXHAUSTIVE",
                "NULL requires an EXHAUSTIVE determination of no active event",
            )
        return cls(None, None, ())


@dataclass(frozen=True, slots=True)
class TopologyMember:
    topology: tuple[ConstraintType, ...]
    payload_ranks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    resource_id: int
    claim_mode: ClaimMode
    target_namespace: TargetNamespace
    target_selector: TargetSelector
    target_key: bytes

    @property
    def resource_target_id(self) -> tuple[int, TargetNamespace, TargetSelector, bytes]:
        return (
            self.resource_id,
            self.target_namespace,
            self.target_selector,
            self.target_key,
        )


@dataclass(frozen=True, slots=True)
class ResourceValidationContext:
    controller_target_enums: tuple[int, ...]
    constant_target_enums: tuple[int, ...]
    actuator_resource_ids: tuple[int, ...]
    compatible_wrist_target_pairs: tuple[
        frozenset[tuple[int, TargetNamespace, TargetSelector, bytes]], ...
    ]
    canonical_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class LocatorEvidence:
    """Reviewed evidence used before constructing a hard locator.

    ``None`` denotes unknown/missing.  Reconstruction/completion is explicit
    and is never accepted as hard-packet evidence.
    """

    kind: LocatorKind
    uv: int | None
    depth: int | None
    axis: int | None = None
    roll: int | None = None
    primitive: PathPrimitive | None = None
    extent: int | None = None
    reviewed_known: bool = False
    expired: bool = False
    support_unique: bool = False
    transparent_overlap: bool = False
    direct_depth: bool = False
    reconstructed: bool = False


@dataclass(frozen=True, slots=True)
class HeldPublicState:
    binding_kind: int
    controlled_part: int
    transform: tuple[float, ...]
    translation_uncertainty_m: float
    angular_uncertainty_rad: float


@dataclass(frozen=True, slots=True)
class PublicExecutionState:
    anchor_timestamp_ns: int
    canonical_proprio: tuple[float, ...]
    camera_mode: int
    camera_transform: tuple[float, ...]
    held: tuple[HeldPublicState | None, HeldPublicState | None]
    canonical_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicStateValidationPolicy:
    quaternion_unit_tolerance: float
    public_state_max_age_ns: int
    verify_position_tolerance_m: float
    verify_axis_tolerance_rad: float
    verify_frame_tolerance_rad: float
    held_binding_kinds: tuple[tuple[int, ...], tuple[int, ...]]
    canonical_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class FrameBasis:
    x: tuple[float, float, float]
    y: tuple[float, float, float]
    z: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RigidTransform:
    translation: tuple[float, float, float]
    rotation: FrameBasis


@dataclass(frozen=True, slots=True)
class GoalProjection:
    kind: LocatorKind
    position: tuple[float, float, float]
    axis: tuple[float, float, float] | None = None
    frame: FrameBasis | None = None


@dataclass(frozen=True, slots=True)
class PathPose:
    parameter: float
    position: tuple[float, float, float]
    frame: FrameBasis


def _strict_int(value: object, code: str, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SchemaError(code, f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _strict_bool(value: object, code: str, name: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(code, f"{name} must be bool")
    return value


def _strict_enum(value: object, enum_type: type[IntEnum], code: str, name: str) -> IntEnum:
    if type(value) is not enum_type:
        raise SchemaError(code, f"{name} must be {enum_type.__name__}, not {value!r}")
    return value


def _constraint_type(constraint: Constraint) -> ConstraintType:
    function = _strict_enum(
        constraint.function,
        ConstraintFunction,
        "FUNCTION_ASSIGNMENT",
        "constraint.function",
    )
    assignment = _strict_enum(
        constraint.assignment,
        Assignment,
        "FUNCTION_ASSIGNMENT",
        "constraint.assignment",
    )
    try:
        return _FIELDS_TYPE[(function, assignment)]
    except KeyError as exc:
        raise SchemaError("FUNCTION_ASSIGNMENT", "function/assignment pair is illegal") from exc


def _validate_locator(locator: Locator, *, verify: bool) -> None:
    if not isinstance(locator, Locator):
        raise SchemaError("FIELD_RANGE_OR_PAD", "locator must be Locator")
    kind = _strict_enum(locator.kind, LocatorKind, "LOCATOR_KIND", "locator.kind")
    if type(locator.uv) is not int or type(locator.depth) is not int:
        raise SchemaError("FIELD_RANGE_OR_PAD", "uv/depth must be integers")
    if locator.uv == 0 or locator.depth == 0:
        raise SchemaError("FIELD_RANGE_OR_PAD", "positive locators require uv and depth")
    _strict_int(locator.uv, "FIELD_RANGE", "locator.uv", minimum=1, maximum=4096)
    _strict_int(locator.depth, "FIELD_RANGE", "locator.depth", minimum=1, maximum=32)
    for name, value, maximum in (
        ("axis", locator.axis, 512),
        ("roll", locator.roll, 32),
        ("extent", locator.extent, 8),
    ):
        _strict_int(value, "FIELD_RANGE", f"locator.{name}", minimum=0, maximum=maximum)
    if locator.primitive != 0 and type(locator.primitive) is not PathPrimitive:
        raise SchemaError("FIELD_RANGE", "locator.primitive must be PathPrimitive or integer PAD=0")
    if locator.primitive == 0 and type(locator.primitive) is not int:
        raise SchemaError("FIELD_RANGE", "locator.primitive PAD must be integer zero")
    if verify and kind is LocatorKind.PATH:
        raise SchemaError("VERIFY_PATH_FORBIDDEN", "VERIFY does not admit PATH")
    canonical = {
        LocatorKind.POINT: locator.axis == locator.roll == locator.primitive == locator.extent == 0,
        LocatorKind.AXIS: 1 <= locator.axis <= 512
        and locator.roll == locator.primitive == locator.extent == 0,
        LocatorKind.FRAME: 1 <= locator.axis <= 512
        and 1 <= locator.roll <= 32
        and locator.primitive == locator.extent == 0,
        LocatorKind.PATH: 1 <= locator.axis <= 512
        and 1 <= locator.roll <= 32
        and 1 <= locator.primitive <= 4
        and 1 <= locator.extent <= 8,
    }[kind]
    if not canonical:
        raise SchemaError("LOCATOR_KIND", "locator fields are noncanonical for its kind")


def _validate_secondary(constraint: Constraint) -> None:
    primary = _strict_enum(
        constraint.primary_role, Role, "FIELD_RANGE", "constraint.primary_role"
    )
    secondary = _strict_enum(
        constraint.secondary_role, Role, "SECONDARY_RELATION", "constraint.secondary_role"
    )
    relation = _strict_enum(
        constraint.relation, Relation, "SECONDARY_RELATION", "constraint.relation"
    )
    if primary is Role.NONE:
        raise SchemaError("FIELD_RANGE_OR_PAD", "primary role is required")
    if (secondary is Role.NONE) != (relation is Relation.N_A):
        raise SchemaError("SECONDARY_RELATION", "secondary NONE iff relation N_A")
    if secondary is not Role.NONE and secondary is primary:
        raise SchemaError("SECONDARY_RELATION", "secondary role must differ from primary")


def _validate_constraint_fields(constraint: Constraint, *, verify: bool) -> None:
    if not isinstance(constraint, Constraint):
        raise SchemaError("FIELD_RANGE_OR_PAD", "constraint must be Constraint")
    primary = _strict_enum(
        constraint.primary_role, Role, "FIELD_RANGE", "constraint.primary_role"
    )
    if primary is Role.NONE:
        raise SchemaError("FIELD_RANGE_OR_PAD", "primary role is required")
    part = _strict_enum(constraint.part, Part, "FIELD_RANGE", "constraint.part")
    if verify:
        if part is not Part.N_A:
            raise SchemaError("FIELD_RANGE", "VERIFY part must be N_A")
    elif part not in tuple(Part(code) for code in range(2, 9)):
        raise SchemaError("FIELD_RANGE", "nonVERIFY part must be one of codes 2..8")
    _validate_locator(constraint.locator, verify=verify)
    _constraint_type(constraint)
    _validate_secondary(constraint)


def validate_packet(packet: EventPacket) -> None:
    """Validate a typed packet in the frozen error-priority order."""

    if not isinstance(packet, EventPacket):
        raise SchemaError("FIELD_RANGE_OR_PAD", "packet must be EventPacket")
    if packet.phase is None or packet.effect is None:
        if packet.phase is not None or packet.effect is not None or packet.constraints:
            raise SchemaError("NULL_CANONICALITY", "NULL is uniquely phase/effect absent and empty")
        return
    phase = _strict_enum(packet.phase, Phase, "HEADER_BLOCK", "packet.phase")
    effect = _strict_enum(packet.effect, Effect, "HEADER_BLOCK", "packet.effect")
    try:
        header_index = _HEADER_TO_INDEX[(phase, effect)]
    except KeyError as exc:
        raise SchemaError("HEADER_BLOCK", "phase/effect pair is not a frozen header block") from exc
    if type(packet.constraints) is not tuple or not 1 <= len(packet.constraints) <= 2:
        raise SchemaError("FUNCTION_ASSIGNMENT", "positive packet requires one or two constraints")
    verify = header_index == 11
    # Fields/locator precede function/bundle and secondary checks.
    for constraint in packet.constraints:
        if not isinstance(constraint, Constraint):
            raise SchemaError("FIELD_RANGE_OR_PAD", "constraint must be Constraint")
        primary = _strict_enum(
            constraint.primary_role, Role, "FIELD_RANGE", "constraint.primary_role"
        )
        if primary is Role.NONE:
            raise SchemaError("FIELD_RANGE_OR_PAD", "primary role is required")
        part = _strict_enum(constraint.part, Part, "FIELD_RANGE", "constraint.part")
        if (verify and part is not Part.N_A) or (
            not verify and part not in tuple(Part(code) for code in range(2, 9))
        ):
            raise SchemaError("FIELD_RANGE", "part is illegal in this header block")
        _validate_locator(constraint.locator, verify=verify)
    types = tuple(_constraint_type(item) for item in packet.constraints)
    topologies = VERIFY_TOPOLOGIES if verify else NONVERIFY_TOPOLOGIES
    if types not in topologies:
        swapped = len(types) == 2 and tuple(reversed(types)) in topologies
        raise SchemaError("SORT" if swapped else "FUNCTION_ASSIGNMENT", "illegal bundle topology")
    for constraint in packet.constraints:
        _validate_secondary(constraint)
    if len(types) == 2 and types[0] is types[1]:
        q0 = payload_rank(packet.constraints[0], verify=verify, validated=True)
        q1 = payload_rank(packet.constraints[1], verify=verify, validated=True)
        if q0 == q1:
            raise SchemaError("DUPLICATE", "same-type payloads must be distinct")
        if q0 > q1:
            raise SchemaError("SORT", "same-type payloads must satisfy q0 < q1")


def _pack_constraint(constraint: Constraint) -> bytes:
    locator = constraint.locator
    return struct.pack(
        "<BBBBBHBHBBBBB",
        int(constraint.function),
        int(constraint.assignment),
        int(constraint.primary_role),
        int(constraint.part),
        int(locator.kind),
        locator.uv,
        locator.depth,
        locator.axis,
        locator.roll,
        locator.primitive,
        locator.extent,
        int(constraint.secondary_role),
        int(constraint.relation),
    )


def pack_wire(packet: EventPacket) -> bytes:
    """Validate and pack the unique canonical 32-byte field wire."""

    validate_packet(packet)
    if packet.phase is None:
        return bytes(WIRE_BYTES)
    constraints = b"".join(_pack_constraint(item) for item in packet.constraints)
    constraints += bytes(CONSTRAINT_BYTES * (2 - len(packet.constraints)))
    return bytes((int(packet.phase), int(packet.effect))) + constraints


def _wire_enum(enum_type: type[IntEnum], value: int, code: str, name: str) -> IntEnum:
    try:
        return enum_type(value)
    except (ValueError, OverflowError) as exc:
        raise SchemaError(code, f"invalid {name} code {value}") from exc


def _unpack_constraint(raw: bytes) -> Constraint:
    (
        function,
        assignment,
        primary,
        part,
        kind,
        uv,
        depth,
        axis,
        roll,
        primitive,
        extent,
        secondary,
        relation,
    ) = struct.unpack("<BBBBBHBHBBBBB", raw)
    return Constraint(
        _wire_enum(ConstraintFunction, function, "FUNCTION_ASSIGNMENT", "function"),
        _wire_enum(Assignment, assignment, "FUNCTION_ASSIGNMENT", "assignment"),
        _wire_enum(Role, primary, "FIELD_RANGE", "primary role"),
        _wire_enum(Part, part, "FIELD_RANGE", "part"),
        Locator(
            _wire_enum(LocatorKind, kind, "LOCATOR_KIND", "locator kind"),
            uv,
            depth,
            axis,
            roll,
            PathPrimitive(primitive) if primitive else 0,
            extent,
        ),
        _wire_enum(Role, secondary, "SECONDARY_RELATION", "secondary role"),
        _wire_enum(Relation, relation, "SECONDARY_RELATION", "relation"),
    )


def _preflight_wire_constraints(raws: Sequence[bytes], *, verify: bool) -> None:
    """Apply raw-byte checks before enum construction, preserving error priority."""

    decoded = [struct.unpack("<BBBBBHBHBBBBB", raw) for raw in raws]
    # FIELD_RANGE/PAD precedes locator, function, and secondary semantics.
    for fields in decoded:
        _, _, primary, part, _, uv, depth, axis, roll, primitive, extent, _, _ = fields
        if not 1 <= primary <= 4:
            code = "FIELD_RANGE_OR_PAD" if primary == 0 else "FIELD_RANGE"
            raise SchemaError(code, "primary role is outside range")
        if (verify and part != int(Part.N_A)) or (not verify and not 2 <= part <= 8):
            code = "FIELD_RANGE_OR_PAD" if part == 0 else "FIELD_RANGE"
            raise SchemaError(code, "part is outside this header alphabet")
        if uv == 0 or depth == 0:
            raise SchemaError("FIELD_RANGE_OR_PAD", "positive locator requires uv/depth")
        if not 1 <= uv <= 4096 or not 1 <= depth <= 32:
            raise SchemaError("FIELD_RANGE", "translation code is outside range")
        if axis > 512 or roll > 32 or primitive > 4 or extent > 8:
            raise SchemaError("FIELD_RANGE", "locator component is outside range")
    # Locator-kind coherence precedes function/assignment.
    for fields in decoded:
        _, _, _, _, kind, _, _, axis, roll, primitive, extent, _, _ = fields
        if not 1 <= kind <= 4:
            raise SchemaError("LOCATOR_KIND", "locator kind code is outside range")
        if verify and kind == int(LocatorKind.PATH):
            raise SchemaError("VERIFY_PATH_FORBIDDEN", "VERIFY does not admit PATH")
        coherent = {
            int(LocatorKind.POINT): axis == roll == primitive == extent == 0,
            int(LocatorKind.AXIS): 1 <= axis <= 512 and roll == primitive == extent == 0,
            int(LocatorKind.FRAME): 1 <= axis <= 512
            and 1 <= roll <= 32
            and primitive == extent == 0,
            int(LocatorKind.PATH): 1 <= axis <= 512
            and 1 <= roll <= 32
            and 1 <= primitive <= 4
            and 1 <= extent <= 8,
        }[kind]
        if not coherent:
            raise SchemaError("LOCATOR_KIND", "locator fields are noncanonical for kind")
    # Function/assignment and topology precede secondary/relation.
    types: list[ConstraintType] = []
    for fields in decoded:
        function, assignment = fields[:2]
        try:
            pair = (ConstraintFunction(function), Assignment(assignment))
            types.append(_FIELDS_TYPE[pair])
        except (ValueError, KeyError) as exc:
            raise SchemaError("FUNCTION_ASSIGNMENT", "function/assignment pair is illegal") from exc
    topologies = VERIFY_TOPOLOGIES if verify else NONVERIFY_TOPOLOGIES
    topology = tuple(types)
    if topology not in topologies:
        swapped = len(topology) == 2 and tuple(reversed(topology)) in topologies
        raise SchemaError("SORT" if swapped else "FUNCTION_ASSIGNMENT", "illegal bundle topology")
    for fields in decoded:
        primary, secondary, relation = fields[2], fields[11], fields[12]
        if secondary > 4 or relation > 7:
            raise SchemaError("SECONDARY_RELATION", "secondary/relation code is outside range")
        if (secondary == 0) != (relation == 0) or (secondary != 0 and secondary == primary):
            raise SchemaError("SECONDARY_RELATION", "secondary/relation pair is illegal")


def unpack_wire(wire: bytes) -> EventPacket:
    """Decode a canonical field wire; noncanonical padding is rejected."""

    if type(wire) is not bytes or len(wire) != WIRE_BYTES:
        raise SchemaError("LENGTH", f"wire must be exactly {WIRE_BYTES} bytes")
    if wire == bytes(WIRE_BYTES):
        return EventPacket(None, None, ())
    if wire[0] == 0 or wire[1] > int(Effect.TRANSFER):
        raise SchemaError("NULL_CANONICALITY" if wire[0] == 0 else "HEADER_BLOCK")
    phase = _wire_enum(Phase, wire[0], "HEADER_BLOCK", "phase")
    effect = _wire_enum(Effect, wire[1], "HEADER_BLOCK", "effect")
    raw_constraints: list[bytes] = []
    saw_pad = False
    for index in range(2):
        start = 2 + index * CONSTRAINT_BYTES
        raw = wire[start : start + CONSTRAINT_BYTES]
        if raw == bytes(CONSTRAINT_BYTES):
            saw_pad = True
            continue
        if saw_pad:
            raise SchemaError("FIELD_RANGE_OR_PAD", "active constraint follows padded slot")
        # Inspect range/padding before enum conversion to preserve error priority.
        if raw[4] == 0 or raw[5:7] == b"\x00\x00" or raw[7] == 0:
            raise SchemaError("FIELD_RANGE_OR_PAD", "positive constraint has PAD in required field")
        raw_constraints.append(raw)
    try:
        header_index = _HEADER_TO_INDEX[(phase, effect)]
    except KeyError as exc:
        raise SchemaError("HEADER_BLOCK", "phase/effect pair is not a frozen header block") from exc
    _preflight_wire_constraints(raw_constraints, verify=header_index == 11)
    constraints = [_unpack_constraint(raw) for raw in raw_constraints]
    packet = EventPacket(phase, effect, tuple(constraints))
    validate_packet(packet)
    if pack_wire(packet) != wire:
        raise SchemaError("SORT", "decoded wire did not re-encode byte-for-byte")
    return packet


def locator_rank(locator: Locator, *, verify: bool) -> int:
    _strict_bool(verify, "LOCATOR_KIND", "verify")
    _validate_locator(locator, verify=verify)
    translation = (locator.uv - 1) * 32 + (locator.depth - 1)
    if locator.kind is LocatorKind.POINT:
        variant = 0
    elif locator.kind is LocatorKind.AXIS:
        variant = 1 + locator.axis - 1
    elif locator.kind is LocatorKind.FRAME:
        variant = 1 + 512 + (locator.axis - 1) * 32 + locator.roll - 1
    else:
        variant = (
            1
            + 512
            + 512 * 32
            + (((locator.axis - 1) * 32 + locator.roll - 1) * 4 + locator.primitive - 1)
            * 8
            + locator.extent
            - 1
        )
    width = VERIFY_VARIANTS if verify else NONVERIFY_VARIANTS
    return translation * width + variant


def unrank_locator(rank: int, *, verify: bool) -> Locator:
    _strict_bool(verify, "INVALID_LOCATOR_RANK", "verify")
    limit = L_VERIFY if verify else L_NONVERIFY
    _strict_int(rank, "INVALID_LOCATOR_RANK", "locator rank", minimum=0, maximum=limit - 1)
    width = VERIFY_VARIANTS if verify else NONVERIFY_VARIANTS
    translation, variant = divmod(rank, width)
    uv_minus_one, depth_minus_one = divmod(translation, 32)
    uv = uv_minus_one + 1
    depth = depth_minus_one + 1
    if variant == 0:
        locator = Locator(LocatorKind.POINT, uv, depth)
    elif variant <= 512:
        locator = Locator(LocatorKind.AXIS, uv, depth, axis=variant)
    elif variant < VERIFY_VARIANTS:
        axis_minus_one, roll_minus_one = divmod(variant - 1 - 512, 32)
        locator = Locator(
            LocatorKind.FRAME,
            uv,
            depth,
            axis=axis_minus_one + 1,
            roll=roll_minus_one + 1,
        )
    else:
        if verify:
            raise SchemaError("VERIFY_PATH_FORBIDDEN", "VERIFY locator rank enters PATH interval")
        high, extent_minus_one = divmod(variant - VERIFY_VARIANTS, 8)
        axis_roll, primitive_minus_one = divmod(high, 4)
        axis_minus_one, roll_minus_one = divmod(axis_roll, 32)
        locator = Locator(
            LocatorKind.PATH,
            uv,
            depth,
            axis=axis_minus_one + 1,
            roll=roll_minus_one + 1,
            primitive=PathPrimitive(primitive_minus_one + 1),
            extent=extent_minus_one + 1,
        )
    if locator_rank(locator, verify=verify) != rank:
        raise SchemaError("INVALID_LOCATOR_RANK", "locator rank has noncanonical residue")
    return locator


def _secondary_rank(constraint: Constraint) -> int:
    _validate_secondary(constraint)
    if constraint.secondary_role is Role.NONE:
        return 0
    roles = [Role.SOURCE, Role.TOOL, Role.DESTINATION, Role.SUPPORT]
    roles.remove(constraint.primary_role)
    return 1 + roles.index(constraint.secondary_role) * 7 + int(constraint.relation) - 1


def _unrank_secondary(rank: int, primary: Role) -> tuple[Role, Relation]:
    _strict_int(rank, "INVALID_PAYLOAD_RANK", "secondary rank", minimum=0, maximum=21)
    if rank == 0:
        return Role.NONE, Relation.N_A
    local, relation_minus_one = divmod(rank - 1, 7)
    roles = [Role.SOURCE, Role.TOOL, Role.DESTINATION, Role.SUPPORT]
    roles.remove(primary)
    if local >= len(roles):
        raise SchemaError("INVALID_PAYLOAD_RANK", "secondary local role is out of range")
    return roles[local], Relation(relation_minus_one + 1)


def payload_rank(constraint: Constraint, *, verify: bool, validated: bool = False) -> int:
    _strict_bool(verify, "INVALID_PAYLOAD_RANK", "verify")
    _strict_bool(validated, "INVALID_PAYLOAD_RANK", "validated")
    if not validated:
        _validate_constraint_fields(constraint, verify=verify)
    part_index = 0 if verify else int(constraint.part) - 2
    primary_index = int(constraint.primary_role) - 1
    location = locator_rank(constraint.locator, verify=verify)
    location_count = L_VERIFY if verify else L_NONVERIFY
    return (
        ((part_index * 4 + primary_index) * location_count + location) * 22
        + _secondary_rank(constraint)
    )


def unrank_payload(rank: int, constraint_type: ConstraintType, *, verify: bool) -> Constraint:
    _strict_bool(verify, "INVALID_PAYLOAD_RANK", "verify")
    if type(constraint_type) is not ConstraintType:
        raise SchemaError("INVALID_PAYLOAD_RANK", "constraint_type must be ConstraintType")
    limit = V if verify else N
    _strict_int(rank, "INVALID_PAYLOAD_RANK", "payload rank", minimum=0, maximum=limit - 1)
    high, secondary_rank = divmod(rank, 22)
    location_count = L_VERIFY if verify else L_NONVERIFY
    part_role, location_rank = divmod(high, location_count)
    part_index, primary_index = divmod(part_role, 4)
    if verify:
        if part_index != 0:
            raise SchemaError("INVALID_PAYLOAD_RANK", "VERIFY part quotient is nonzero")
        part = Part.N_A
    else:
        if not 0 <= part_index < 7:
            raise SchemaError("INVALID_PAYLOAD_RANK", "part quotient is out of range")
        part = Part(part_index + 2)
    if not 0 <= primary_index < 4:
        raise SchemaError("INVALID_PAYLOAD_RANK", "primary role quotient is out of range")
    primary = Role(primary_index + 1)
    secondary, relation = _unrank_secondary(secondary_rank, primary)
    try:
        function, assignment = _TYPE_FIELDS[constraint_type]
    except KeyError as exc:
        raise SchemaError("INVALID_PAYLOAD_RANK", "unknown constraint type") from exc
    constraint = Constraint(
        function,
        assignment,
        primary,
        part,
        unrank_locator(location_rank, verify=verify),
        secondary,
        relation,
    )
    if payload_rank(constraint, verify=verify) != rank:
        raise SchemaError("INVALID_PAYLOAD_RANK", "payload rank did not round-trip")
    return constraint


def combination_rank(q0: int, q1: int, cardinality: int) -> int:
    _strict_int(cardinality, "INVALID_COMB_RANK", "cardinality", minimum=2, maximum=MAX_GLOBAL_RANK)
    _strict_int(q0, "INVALID_COMB_RANK", "q0", minimum=0, maximum=cardinality - 1)
    _strict_int(q1, "INVALID_COMB_RANK", "q1", minimum=0, maximum=cardinality - 1)
    if q0 >= q1:
        raise SchemaError("INVALID_COMB_RANK", "combination requires q0 < q1")
    return q0 * (2 * cardinality - q0 - 1) // 2 + q1 - q0 - 1


def unrank_combination(rank: int, cardinality: int) -> tuple[int, int]:
    _strict_int(cardinality, "INVALID_COMB_RANK", "cardinality", minimum=2, maximum=MAX_GLOBAL_RANK)
    total = cardinality * (cardinality - 1) // 2
    _strict_int(rank, "INVALID_COMB_RANK", "combination rank", minimum=0, maximum=total - 1)

    def start(index: int) -> int:
        return index * (2 * cardinality - index - 1) // 2

    low, high = 0, cardinality - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if start(middle) <= rank:
            low = middle
        else:
            high = middle
    q0 = low
    q1 = q0 + 1 + rank - start(q0)
    if not q0 < q1 < cardinality or combination_rank(q0, q1, cardinality) != rank:
        raise SchemaError("INVALID_COMB_RANK", "integer combinadic inverse failed")
    return q0, q1


def topology_capacity(*, verify: bool, cardinality: int) -> int:
    _strict_bool(verify, "INVALID_PAYLOAD_RANK", "verify")
    _strict_int(cardinality, "INVALID_PAYLOAD_RANK", "cardinality", minimum=2, maximum=N)
    if verify:
        return 3 * cardinality + 3 * cardinality * cardinality + cardinality * (cardinality - 1)
    return 2 * cardinality + 7 * cardinality * cardinality + cardinality * (cardinality - 1) // 2


def rank_topology(member: TopologyMember, *, verify: bool, cardinality: int) -> int:
    _strict_bool(verify, "INVALID_PAYLOAD_RANK", "verify")
    topologies = VERIFY_TOPOLOGIES if verify else NONVERIFY_TOPOLOGIES
    if not isinstance(member, TopologyMember) or member.topology not in topologies:
        raise SchemaError("FUNCTION_ASSIGNMENT", "topology is not admitted")
    if any(type(item) is not ConstraintType for item in member.topology):
        raise SchemaError("FUNCTION_ASSIGNMENT", "topology members must be ConstraintType")
    if len(member.payload_ranks) != len(member.topology):
        raise SchemaError("INVALID_PAYLOAD_RANK", "payload count differs from topology")
    for rank in member.payload_ranks:
        _strict_int(rank, "INVALID_PAYLOAD_RANK", "payload rank", minimum=0, maximum=cardinality - 1)
    offset = 0
    for topology in topologies:
        same_type = len(topology) == 2 and topology[0] is topology[1]
        size = (
            cardinality
            if len(topology) == 1
            else cardinality * (cardinality - 1) // 2
            if same_type
            else cardinality * cardinality
        )
        if topology == member.topology:
            if len(topology) == 1:
                local = member.payload_ranks[0]
            elif same_type:
                local = combination_rank(*member.payload_ranks, cardinality)
            else:
                local = member.payload_ranks[0] * cardinality + member.payload_ranks[1]
            return offset + local
        offset += size
    raise SchemaError("FUNCTION_ASSIGNMENT", "topology interval was not found")


def unrank_topology(rank: int, *, verify: bool, cardinality: int) -> TopologyMember:
    _strict_bool(verify, "INVALID_PAYLOAD_RANK", "verify")
    capacity = topology_capacity(verify=verify, cardinality=cardinality)
    _strict_int(rank, "INVALID_PAYLOAD_RANK", "bundle rank", minimum=0, maximum=capacity - 1)
    topologies = VERIFY_TOPOLOGIES if verify else NONVERIFY_TOPOLOGIES
    offset = 0
    for topology in topologies:
        same_type = len(topology) == 2 and topology[0] is topology[1]
        size = (
            cardinality
            if len(topology) == 1
            else cardinality * (cardinality - 1) // 2
            if same_type
            else cardinality * cardinality
        )
        if rank < offset + size:
            local = rank - offset
            if len(topology) == 1:
                payloads = (local,)
            elif same_type:
                payloads = unrank_combination(local, cardinality)
            else:
                payloads = divmod(local, cardinality)
            member = TopologyMember(topology, payloads)
            if rank_topology(member, verify=verify, cardinality=cardinality) != rank:
                raise SchemaError("INVALID_PAYLOAD_RANK", "bundle rank did not round-trip")
            return member
        offset += size
    raise SchemaError("INVALID_PAYLOAD_RANK", "bundle rank interval was not found")


def rank_packet(packet: EventPacket) -> int:
    validate_packet(packet)
    if packet.phase is None:
        return 0
    header = _HEADER_TO_INDEX[(packet.phase, packet.effect)]
    verify = header == 11
    types = tuple(_constraint_type(item) for item in packet.constraints)
    payloads = tuple(payload_rank(item, verify=verify, validated=True) for item in packet.constraints)
    bundle = rank_topology(
        TopologyMember(types, payloads),
        verify=verify,
        cardinality=V if verify else N,
    )
    if verify:
        return 1 + 11 * B_NONVERIFY + bundle
    return 1 + header * B_NONVERIFY + bundle


def rank_wire(wire: bytes) -> int:
    return rank_packet(unpack_wire(wire))


def unrank_packet(rank: int) -> EventPacket:
    _strict_int(
        rank,
        "INVALID_GLOBAL_RANK",
        "global rank",
        minimum=0,
        maximum=STRUCTURAL_STATES - 1,
    )
    if rank == 0:
        return EventPacket(None, None, ())
    residual = rank - 1
    if residual < 11 * B_NONVERIFY:
        header, bundle = divmod(residual, B_NONVERIFY)
        verify = False
        cardinality = N
    else:
        header = 11
        bundle = residual - 11 * B_NONVERIFY
        verify = True
        cardinality = V
    member = unrank_topology(bundle, verify=verify, cardinality=cardinality)
    constraints = tuple(
        unrank_payload(payload, constraint_type, verify=verify)
        for constraint_type, payload in zip(member.topology, member.payload_ranks, strict=True)
    )
    phase, effect = HEADER_BLOCKS[header]
    packet = EventPacket(phase, effect, constraints)
    if rank_packet(packet) != rank:
        raise SchemaError("INVALID_GLOBAL_RANK", "global unrank did not round-trip")
    return packet


def unrank_wire(rank: int) -> bytes:
    wire = pack_wire(unrank_packet(rank))
    if rank_wire(wire) != rank:
        raise SchemaError("INVALID_GLOBAL_RANK", "global wire unrank did not round-trip")
    return wire


def pack_compact(rank: int) -> bytes:
    _strict_int(
        rank,
        "INVALID_GLOBAL_RANK",
        "global rank",
        minimum=0,
        maximum=STRUCTURAL_STATES - 1,
    )
    return rank.to_bytes(COMPACT_BYTES, "little", signed=False)


def unpack_compact(container: bytes) -> int:
    if type(container) is not bytes or len(container) != COMPACT_BYTES:
        raise SchemaError("LENGTH", f"compact container must be exactly {COMPACT_BYTES} bytes")
    if container[12] & 0xFE:
        raise SchemaError("RESERVED_BITS", "bits 97..103 must be zero")
    rank = int.from_bytes(container, "little", signed=False)
    if rank >= STRUCTURAL_STATES:
        raise SchemaError("INVALID_GLOBAL_RANK", "compact integer is outside structural alphabet")
    return rank


def field_wire_to_compact(wire: bytes) -> bytes:
    return pack_compact(rank_wire(wire))


def compact_to_field_wire(container: bytes) -> bytes:
    return unrank_wire(unpack_compact(container))


def equal_bit_permute(rank: int) -> int:
    _strict_int(rank, "INVALID_GLOBAL_RANK", "global rank", minimum=0, maximum=MAX_GLOBAL_RANK)
    if rank == 0:
        return 0
    return 1 + (EQUAL_BIT_MULTIPLIER * (rank - 1) + EQUAL_BIT_OFFSET) % MAX_GLOBAL_RANK


def equal_bit_inverse(rank: int) -> int:
    _strict_int(rank, "INVALID_GLOBAL_RANK", "permuted rank", minimum=0, maximum=MAX_GLOBAL_RANK)
    if rank == 0:
        return 0
    return 1 + (
        EQUAL_BIT_INVERSE * ((rank - 1 - EQUAL_BIT_OFFSET) % MAX_GLOBAL_RANK)
    ) % MAX_GLOBAL_RANK


def capacity_metadata() -> dict[str, int | str]:
    """Return exact structural constants, never executable-profile claims."""

    return {
        "schema_id": SCHEMA_ID,
        "wire_bytes": WIRE_BYTES,
        "hard_token_slots": HARD_TOKEN_SLOTS,
        "translations": TRANSLATIONS,
        "locator_states_nonverify": L_NONVERIFY,
        "locator_states_verify": L_VERIFY,
        "payload_states_nonverify": N,
        "payload_states_verify": V,
        "bundle_states_nonverify": B_NONVERIFY,
        "bundle_states_verify": B_VERIFY,
        "structural_states": STRUCTURAL_STATES,
        "log2_structural_states": STRUCTURAL_LOG2,
        "minimum_index_bits": MINIMUM_INDEX_BITS,
        "compact_bytes": COMPACT_BYTES,
        "g1_profile_states": "UNDEFINED / NO-GO",
        "g1_profile_bits": "UNDEFINED / NO-GO",
    }


def profile_bundle_capacity(
    *,
    verify: bool,
    sizes: Mapping[str, int],
    compatible_pairs: Mapping[str, int],
    compatible_unordered: Mapping[str, int],
) -> int:
    """Evaluate the frozen finite-profile formula from already enumerated sets."""

    if type(verify) is not bool:
        raise SchemaError("PROFILE_COUNT", "verify must be bool")
    expected_sizes = {"LS", "RS", "ZM"} if verify else {"LO", "LS", "RO", "RS", "ZM"}
    expected_pairs = (
        {"LS-RS", "LS-ZM", "RS-ZM"}
        if verify
        else {"LO-RO", "LO-RS", "LS-RO", "LO-ZM", "RO-ZM", "LO-LS", "RO-RS"}
    )
    expected_unordered = {"ZM", "RS"} if verify else {"RO"}
    for mapping, expected, name in (
        (sizes, expected_sizes, "sizes"),
        (compatible_pairs, expected_pairs, "compatible_pairs"),
        (compatible_unordered, expected_unordered, "compatible_unordered"),
    ):
        if type(mapping) is not dict or set(mapping) != expected:
            raise SchemaError("PROFILE_COUNT", f"{name} must have exact keys {sorted(expected)}")
        for key, value in mapping.items():
            _strict_int(value, "PROFILE_COUNT", f"{name}.{key}", minimum=0, maximum=MAX_GLOBAL_RANK)
    for pair, count in compatible_pairs.items():
        left, right = pair.split("-", 1)
        maximum = sizes[left] * sizes[right]
        if count > maximum:
            raise SchemaError(
                "PROFILE_COUNT",
                f"C({left},{right})={count} exceeds |{left}|*|{right}|={maximum}",
            )
    for name, count in compatible_unordered.items():
        maximum = sizes[name] * (sizes[name] - 1) // 2
        if count > maximum:
            raise SchemaError(
                "PROFILE_COUNT",
                f"U({name})={count} exceeds C(|{name}|,2)={maximum}",
            )
    if verify:
        return (
            sizes["LS"]
            + sizes["RS"]
            + sizes["ZM"]
            + sum(compatible_pairs.values())
            + sum(compatible_unordered.values())
        )
    return (
        sizes["LO"]
        + sizes["RO"]
        + sum(compatible_pairs.values())
        + compatible_unordered["RO"]
    )


def actuator_ownership_mask(constraint: Constraint) -> tuple[bool, ...]:
    """Return the frozen 25-D split ownership mask; MONITOR is all zero."""

    constraint_type = _constraint_type(constraint)
    mask = [False] * 25
    if constraint_type in (ConstraintType.LO, ConstraintType.LS):
        for index in (*range(0, 9), *range(18, 24)):
            mask[index] = True
    elif constraint_type in (ConstraintType.RO, ConstraintType.RS):
        for index in (*range(9, 18), 24):
            mask[index] = True
    return tuple(mask)


def validate_locator_evidence(evidence: LocatorEvidence, *, relation: Relation) -> Locator:
    """Reject missing/expired/completed or non-identifiable transparent evidence."""

    if not isinstance(evidence, LocatorEvidence):
        raise SchemaError("MISSING_REQUIRED", "evidence must be LocatorEvidence")
    _strict_enum(evidence.kind, LocatorKind, "LOCATOR_KIND", "evidence.kind")
    _strict_enum(relation, Relation, "SECONDARY_RELATION", "relation")
    for name in (
        "reviewed_known",
        "expired",
        "support_unique",
        "transparent_overlap",
        "direct_depth",
        "reconstructed",
    ):
        if type(getattr(evidence, name)) is not bool:
            raise SchemaError("MISSING_REQUIRED", f"{name} must be bool")
    if not evidence.reviewed_known or evidence.expired:
        raise SchemaError("MISSING_REQUIRED", "required locator is unknown, missing, or expired")
    if evidence.reconstructed:
        raise SchemaError("MISSING_REQUIRED", "completion/reconstruction cannot fill hard fields")
    if not evidence.support_unique:
        raise SchemaError("MISSING_REQUIRED", "support crosses non-equivalent safe-action bins")
    if evidence.transparent_overlap:
        if not evidence.direct_depth:
            raise SchemaError("TRANSPARENT_UNOBSERVABLE", "transparent overlap has no direct depth")
        if evidence.kind is not LocatorKind.POINT:
            raise SchemaError("TRANSPARENT_UNOBSERVABLE", "transparent overlap cannot supply normal")
        if relation in (Relation.TOUCHING, Relation.INSIDE):
            raise SchemaError("TRANSPARENT_UNOBSERVABLE", "transparent overlap cannot prove relation")
    required = [evidence.uv, evidence.depth]
    if evidence.kind in (LocatorKind.AXIS, LocatorKind.FRAME, LocatorKind.PATH):
        required.append(evidence.axis)
    if evidence.kind in (LocatorKind.FRAME, LocatorKind.PATH):
        required.append(evidence.roll)
    if evidence.kind is LocatorKind.PATH:
        required.extend((evidence.primitive, evidence.extent))
    if any(value is None for value in required):
        raise SchemaError("MISSING_REQUIRED", "locator kind has an unknown required field")
    locator = Locator(
        evidence.kind,
        evidence.uv if evidence.uv is not None else 0,
        evidence.depth if evidence.depth is not None else 0,
        evidence.axis or 0,
        evidence.roll or 0,
        evidence.primitive or 0,
        evidence.extent or 0,
    )
    _validate_locator(locator, verify=False)
    return locator


_TASK_FAMILIES = {"POUR", "INSERT", "PIPETTE", "CAP", "WIPE"}
_PHASE_NAMES = {item.name for item in Phase}
_EFFECT_NAMES = {item.name for item in Effect}
_SEMANTIC_ROLES = {
    "MANIPULATED_OR_SOURCE",
    "TOOL",
    "RECEPTACLE_OR_DESTINATION",
    "FIXTURE_OR_SUPPORT",
}
_SEMANTIC_PARTS = {
    "BODY",
    "HANDLE_OR_GRASP_FEATURE",
    "LID_CAP",
    "OPENING_OR_RIM",
    "TOOL_TIP",
    "ACTUATOR_OR_PLUNGER",
    "WORKING_FACE_OR_PAD",
    "N_A",
}
_SEMANTIC_RELATIONS = {item.name for item in Relation}
_PURPOSES = {"EFFECT_GOAL", "SUPPORT_REQUIREMENT", "OBSERVATION_REQUIREMENT"}
_PURPOSE_ORDER = {"EFFECT_GOAL": 1, "SUPPORT_REQUIREMENT": 2, "OBSERVATION_REQUIREMENT": 3}
_ROLE_ORDER = {
    "NONE": 0,
    "MANIPULATED_OR_SOURCE": 1,
    "TOOL": 2,
    "RECEPTACLE_OR_DESTINATION": 3,
    "FIXTURE_OR_SUPPORT": 4,
}
_PART_ORDER = {
    "BODY": 1,
    "HANDLE_OR_GRASP_FEATURE": 2,
    "LID_CAP": 3,
    "OPENING_OR_RIM": 4,
    "TOOL_TIP": 5,
    "ACTUATOR_OR_PLUNGER": 6,
    "WORKING_FACE_OR_PAD": 7,
    "N_A": 8,
}
_RELATION_ORDER = {item.name: int(item) for item in Relation}


def canonical_json_bytes(value: object) -> bytes:
    """RFC-8785-equivalent bytes for the schema's integer/ASCII-only domain."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaError("JSON_TYPE", "value is outside the canonical JSON domain") from exc


def parse_strict_json_bytes(raw: bytes) -> object:
    """Parse UTF-8 JSON while rejecting duplicate properties at every depth."""

    if type(raw) is not bytes:
        raise SchemaError("JSON_TYPE", "strict JSON input must be bytes")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError("JSON_DUPLICATE_KEY", f"duplicate object property {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise SchemaError("JSON_TYPE", f"non-finite JSON constant {value!r}")

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject_constant)
    except SchemaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("JSON_TYPE", "invalid UTF-8 JSON bytes") from exc


def _exact_dict(value: object, keys: set[str], *, prefix: str) -> dict[str, Any]:
    if type(value) is bytes:
        value = parse_strict_json_bytes(value)
    if type(value) is not dict:
        raise SchemaError("JSON_TYPE", f"{prefix} must be a JSON object")
    actual = set(value)
    if actual != keys or any(type(key) is not str for key in value):
        extra = sorted(str(key) for key in actual - keys)
        missing = sorted(keys - actual)
        raise SchemaError("JSON_EXTRA_KEY", f"{prefix} extra={extra}, missing={missing}")
    return value


def _json_enum(value: object, allowed: set[str], *, name: str) -> str:
    if type(value) is not str or value not in allowed:
        raise SchemaError("JSON_ENUM", f"{name} is not one complete allowed string")
    return value


def _identifier(value: object, *, name: str, code: str = "JSON_ENUM") -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SchemaError(code, f"{name} is not a frozen ASCII identifier")
    return value


def validate_shared_semantic_key(
    value: object,
    *,
    exhaustive_no_active: bool = False,
) -> bytes:
    """Validate and canonicalize the exact human-shared semantic JSON."""

    top_keys = {
        "schema_id",
        "task_ontology_id",
        "task_family",
        "phase",
        "effect",
        "event_focus_role",
        "constraint_roster_coverage",
        "constraints",
    }
    obj = _exact_dict(value, top_keys, prefix="semantic key")
    if obj["schema_id"] != SHARED_SEMANTIC_SCHEMA_ID:
        raise SchemaError("JSON_ENUM", "semantic schema_id mismatch")
    _identifier(obj["task_ontology_id"], name="task_ontology_id")
    _json_enum(obj["task_family"], _TASK_FAMILIES, name="task_family")
    if obj["constraint_roster_coverage"] != "EXHAUSTIVE":
        raise SchemaError("NULL_NOT_EXHAUSTIVE", "constraint roster must be EXHAUSTIVE")
    constraints = obj["constraints"]
    if type(constraints) is not list:
        raise SchemaError("JSON_TYPE", "constraints must be an array")
    if obj["phase"] == "NULL":
        if type(exhaustive_no_active) is not bool or not exhaustive_no_active:
            raise SchemaError("NULL_NOT_EXHAUSTIVE", "NULL requires reviewed exhaustive no-active")
        expected = {
            "phase": "NULL",
            "effect": "N_A",
            "event_focus_role": "NONE",
        }
        if any(obj[key] != expected[key] for key in expected) or constraints != []:
            raise SchemaError("NULL_NOT_EXHAUSTIVE", "NULL branch fields are noncanonical")
        return canonical_json_bytes(obj)
    _json_enum(obj["phase"], _PHASE_NAMES, name="phase")
    _json_enum(obj["effect"], _EFFECT_NAMES, name="effect")
    _json_enum(obj["event_focus_role"], _SEMANTIC_ROLES, name="event_focus_role")
    try:
        phase = Phase[obj["phase"]]
        effect = Effect[obj["effect"]]
        header = _HEADER_TO_INDEX[(phase, effect)]
    except (KeyError, ValueError) as exc:
        raise SchemaError("HEADER_BLOCK", "semantic phase/effect block is illegal") from exc
    if not 1 <= len(constraints) <= 2:
        raise SchemaError("JSON_TYPE", "nonNULL constraints must have length 1..2")
    constraint_keys = {
        "purpose",
        "primary_role",
        "primary_part",
        "secondary_role",
        "target_relation",
    }
    tuples: list[tuple[int, int, int, int, int]] = []
    purposes: list[str] = []
    primaries: list[str] = []
    for index, raw in enumerate(constraints):
        item = _exact_dict(raw, constraint_keys, prefix=f"constraints[{index}]")
        purpose = _json_enum(item["purpose"], _PURPOSES, name="purpose")
        primary = _json_enum(item["primary_role"], _SEMANTIC_ROLES, name="primary_role")
        part = _json_enum(item["primary_part"], _SEMANTIC_PARTS, name="primary_part")
        secondary = _json_enum(
            item["secondary_role"], _SEMANTIC_ROLES | {"NONE"}, name="secondary_role"
        )
        relation = _json_enum(
            item["target_relation"], _SEMANTIC_RELATIONS, name="target_relation"
        )
        if (secondary == "NONE") != (relation == "N_A") or secondary == primary:
            raise SchemaError("SECONDARY_RELATION", "semantic secondary/relation is illegal")
        if header == 11 and part != "N_A":
            raise SchemaError("JSON_ENUM", "VERIFY semantic part must be N_A")
        if header != 11 and part == "N_A":
            raise SchemaError("JSON_ENUM", "nonVERIFY semantic part must be executable")
        tuples.append(
            (
                _PURPOSE_ORDER[purpose],
                _ROLE_ORDER[primary],
                _PART_ORDER[part],
                _ROLE_ORDER[secondary],
                _RELATION_ORDER[relation],
            )
        )
        purposes.append(purpose)
        primaries.append(primary)
    if len(set(tuples)) != len(tuples):
        raise SchemaError("JSON_DUPLICATE", "exact duplicate semantic constraint")
    if tuples != sorted(tuples):
        raise SchemaError("JSON_SORT", "constraints array is not in frozen ordinal order")
    if header != 11 and "EFFECT_GOAL" not in purposes:
        raise SchemaError("JSON_FOCUS", "nonVERIFY requires an EFFECT_GOAL")
    focus_index = purposes.index("EFFECT_GOAL") if "EFFECT_GOAL" in purposes else 0
    if obj["event_focus_role"] != primaries[focus_index]:
        raise SchemaError("JSON_FOCUS", "event focus differs from mechanical derivation")
    return canonical_json_bytes(obj)


def shared_semantic_key_hash(value: object, *, exhaustive_no_active: bool = False) -> str:
    return hashlib.sha256(
        validate_shared_semantic_key(value, exhaustive_no_active=exhaustive_no_active)
    ).hexdigest()


def validate_shared_semantic_key_json(
    raw: bytes,
    *,
    exhaustive_no_active: bool = False,
) -> bytes:
    return validate_shared_semantic_key(
        parse_strict_json_bytes(raw),
        exhaustive_no_active=exhaustive_no_active,
    )


def shared_semantic_key_hash_json(
    raw: bytes,
    *,
    exhaustive_no_active: bool = False,
) -> str:
    return hashlib.sha256(
        validate_shared_semantic_key_json(raw, exhaustive_no_active=exhaustive_no_active)
    ).hexdigest()


def validate_text_goal(value: object, *, ontology_registry: Mapping[str, str]) -> bytes:
    """Validate text against an immutable ``symbol -> resolver kind`` registry.

    The only admitted registry value is the exact string ``SEMANTIC_ONLY``.
    Locator/trajectory/controller resolvers therefore fail closed.
    """

    keys = {
        "schema_id",
        "task_family",
        "goal_symbol",
        "quantity_class",
        "signed_direction",
        "success_criterion",
    }
    obj = _exact_dict(value, keys, prefix="text goal")
    if obj["schema_id"] != TEXT_GOAL_SCHEMA_ID:
        raise SchemaError("JSON_ENUM", "text schema_id mismatch")
    _json_enum(obj["task_family"], _TASK_FAMILIES, name="task_family")
    symbol = _identifier(obj["goal_symbol"], name="goal_symbol")
    _json_enum(obj["quantity_class"], {"N_A", "LOW", "MEDIUM", "HIGH"}, name="quantity_class")
    _json_enum(obj["signed_direction"], {"N_A", "POSITIVE", "NEGATIVE"}, name="signed_direction")
    _json_enum(
        obj["success_criterion"],
        {"RELATION", "DOSE", "TORQUE", "CLEAN", "POSE"},
        name="success_criterion",
    )
    if not isinstance(ontology_registry, Mapping) or symbol not in ontology_registry:
        raise SchemaError("TEXT_UNKNOWN_GOAL", "goal is absent from immutable registry")
    resolver = ontology_registry[symbol]
    if type(resolver) is not str or resolver != "SEMANTIC_ONLY":
        raise SchemaError("TEXT_LOCATOR_ALIAS", "goal resolves outside the semantic-only domain")
    return canonical_json_bytes(obj)


def text_goal_hash(value: object, *, ontology_registry: Mapping[str, str]) -> str:
    return hashlib.sha256(validate_text_goal(value, ontology_registry=ontology_registry)).hexdigest()


def validate_text_goal_json(raw: bytes, *, ontology_registry: Mapping[str, str]) -> bytes:
    return validate_text_goal(parse_strict_json_bytes(raw), ontology_registry=ontology_registry)


def text_goal_hash_json(raw: bytes, *, ontology_registry: Mapping[str, str]) -> str:
    return hashlib.sha256(
        validate_text_goal_json(raw, ontology_registry=ontology_registry)
    ).hexdigest()


def _float32_values(raw: bytes, count: int, *, context: str) -> tuple[float, ...]:
    if len(raw) != count * 4:
        raise SchemaError("LENGTH", f"{context} float payload has wrong length")
    values = struct.unpack(f"<{count}f", raw)
    for index, (value, bits) in enumerate(zip(values, struct.unpack(f"<{count}I", raw), strict=True)):
        if not math.isfinite(value):
            raise SchemaError("STATE_NONFINITE", f"{context}[{index}] is NaN/Inf")
        if bits == 0x80000000:
            raise SchemaError("STATE_NEGATIVE_ZERO", f"{context}[{index}] is negative zero")
    return values


def _validate_quaternion(values: Sequence[float], tolerance: float, *, context: str) -> None:
    if len(values) != 4:
        raise SchemaError("STATE_QUATERNION_CANONICAL", f"{context} is not length 4")
    if abs(math.sqrt(sum(value * value for value in values)) - 1.0) > tolerance:
        raise SchemaError("STATE_QUATERNION_CANONICAL", f"{context} is not unit within tolerance")
    w, x, y, z = values
    if w < 0.0 or (w == 0.0 and next((v for v in (x, y, z) if v != 0.0), 0.0) <= 0.0):
        raise SchemaError("STATE_QUATERNION_CANONICAL", f"{context} sign is noncanonical")


_PUBLIC_STATE_POLICY_KEYS = {
    "schema_id",
    "quaternion_unit_tolerance_hex",
    "public_state_max_age_ns",
    "verify_position_tolerance_m_hex",
    "verify_axis_tolerance_rad_hex",
    "verify_frame_tolerance_rad_hex",
    "left_held_binding_kinds",
    "right_held_binding_kinds",
}


def _positive_canonical_float_hex(value: object, *, name: str) -> float:
    if type(value) is not str or _FLOAT_HEX_RE.fullmatch(value) is None:
        raise SchemaError("VALIDATION_POLICY", f"{name} must be canonical binary64 hex")
    try:
        decoded = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise SchemaError("VALIDATION_POLICY", f"{name} is malformed") from exc
    if not math.isfinite(decoded) or decoded <= 0.0 or decoded.hex() != value:
        raise SchemaError("VALIDATION_POLICY", f"{name} is not canonical finite positive binary64")
    return decoded


def validate_public_state_validation_policy(
    raw: bytes,
    *,
    expected_sha256: str,
) -> PublicStateValidationPolicy:
    """Validate the immutable policy artifact bound by a static profile receipt."""

    obj = _exact_dict(
        parse_strict_json_bytes(raw),
        _PUBLIC_STATE_POLICY_KEYS,
        prefix="public state validation policy",
    )
    if obj["schema_id"] != "sm_fet_r1d_public_state_validation_policy_v1":
        raise SchemaError("VALIDATION_POLICY", "validation policy schema mismatch")
    canonical = canonical_json_bytes(obj)
    digest = hashlib.sha256(canonical).hexdigest()
    _sha_field(expected_sha256, name="expected validation policy hash", code="VALIDATION_POLICY")
    if digest != expected_sha256:
        raise SchemaError("VALIDATION_POLICY", "validation policy hash mismatch")
    tolerance_names = (
        "quaternion_unit_tolerance_hex",
        "verify_position_tolerance_m_hex",
        "verify_axis_tolerance_rad_hex",
        "verify_frame_tolerance_rad_hex",
    )
    tolerances = tuple(_positive_canonical_float_hex(obj[name], name=name) for name in tolerance_names)
    max_age_ns = _strict_int(
        obj["public_state_max_age_ns"],
        "VALIDATION_POLICY",
        "public_state_max_age_ns",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    kind_lists: list[tuple[int, ...]] = []
    for name in ("left_held_binding_kinds", "right_held_binding_kinds"):
        values = obj[name]
        if type(values) is not list:
            raise SchemaError("VALIDATION_POLICY", f"{name} must be a sorted integer array")
        parsed = tuple(
            _strict_int(value, "VALIDATION_POLICY", name, minimum=1, maximum=255)
            for value in values
        )
        if parsed != tuple(sorted(set(parsed))):
            raise SchemaError("VALIDATION_POLICY", f"{name} must be sorted and duplicate-free")
        kind_lists.append(parsed)
    return PublicStateValidationPolicy(
        tolerances[0],
        max_age_ns,
        tolerances[1],
        tolerances[2],
        tolerances[3],
        (kind_lists[0], kind_lists[1]),
        canonical,
        digest,
    )


def _resolve_bound_validation_policy(
    validation_policy: PublicStateValidationPolicy,
    profile_receipt: object,
) -> PublicStateValidationPolicy:
    if type(validation_policy) is not PublicStateValidationPolicy:
        raise SchemaError("VALIDATION_POLICY", "validation requires a validated policy")
    profile = _validated_profile_receipt_object(profile_receipt)
    if validation_policy.sha256 != profile["public_state_validation_policy_sha256"]:
        raise SchemaError("VALIDATION_POLICY", "policy is not bound by static profile receipt")
    return validate_public_state_validation_policy(
        validation_policy.canonical_bytes,
        expected_sha256=profile["public_state_validation_policy_sha256"],
    )


def validate_public_execution_state(
    state_bytes: bytes,
    *,
    now_ns: int,
    validation_policy: PublicStateValidationPolicy,
    profile_receipt: object,
    expected_sha256: str | None = None,
) -> PublicExecutionState:
    """Validate and parse exact ``public_execution_state_v1`` binary bytes."""

    if type(state_bytes) is not bytes or len(state_bytes) != PUBLIC_STATE_BYTES:
        raise SchemaError("LENGTH", f"public state must be exactly {PUBLIC_STATE_BYTES} bytes")
    if state_bytes[:9] != PUBLIC_STATE_MAGIC or struct.unpack_from("<H", state_bytes, 9)[0] != 1:
        raise SchemaError("STATE_PAD", "public state magic/version mismatch")
    _strict_int(now_ns, "STATE_STALE", "now_ns", minimum=-(1 << 63), maximum=(1 << 63) - 1)
    validation_policy = _resolve_bound_validation_policy(validation_policy, profile_receipt)
    tolerance = validation_policy.quaternion_unit_tolerance
    anchor = struct.unpack_from("<q", state_bytes, 11)[0]
    if anchor > now_ns or now_ns - anchor > validation_policy.public_state_max_age_ns:
        raise SchemaError("STATE_STALE", "anchor timestamp is future or stale")
    offset = 19
    proprio = _float32_values(state_bytes[offset : offset + 100], 25, context="canonical_proprio")
    offset += 100
    camera_mode = state_bytes[offset]
    offset += 1
    if camera_mode not in (0, 1):
        raise SchemaError("STATE_CAMERA_MODE", "camera mode must be FIXED=0 or MOVING=1")
    camera_raw = state_bytes[offset : offset + 28]
    camera = _float32_values(camera_raw, 7, context="T_base_camera")
    offset += 28
    if camera_mode == 0:
        if camera_raw != bytes(28):
            raise SchemaError("STATE_CAMERA_MODE", "FIXED camera transform payload must be positive-zero")
    else:
        _validate_quaternion(camera[3:7], tolerance, context="T_base_camera quaternion")
    held_values: list[HeldPublicState | None] = []
    assignments = (Assignment.LEFT_RH56, Assignment.RIGHT_OMNIPICKER)
    for assignment_index, assignment in enumerate(assignments):
        start = offset
        valid = state_bytes[offset]
        offset += 1
        if valid not in (0, 1):
            raise SchemaError("STATE_PAD", "held_valid must be 0 or 1")
        binding_kind = state_bytes[offset]
        controlled_part = state_bytes[offset + 1]
        offset += 2
        transform = _float32_values(
            state_bytes[offset : offset + 28], 7, context=f"{assignment.name}.held_transform"
        )
        offset += 28
        uncertainties = _float32_values(
            state_bytes[offset : offset + 8], 2, context=f"{assignment.name}.uncertainty"
        )
        offset += 8
        if valid == 0:
            if state_bytes[start + 1 : offset] != bytes(38):
                raise SchemaError("STATE_PAD", "invalid held entry payload must be positive-zero")
            held_values.append(None)
            continue
        allowed = validation_policy.held_binding_kinds[assignment_index]
        if binding_kind not in allowed:
            raise SchemaError("STATE_PAD", "held binding kind is not in immutable profile enum")
        if controlled_part not in range(2, 9):
            raise SchemaError("STATE_PAD", "valid held entry requires controlled part code 2..8")
        _validate_quaternion(transform[3:7], tolerance, context=f"{assignment.name}.held quaternion")
        if any(value < 0.0 for value in uncertainties):
            raise SchemaError("STATE_PAD", "held uncertainties must be nonnegative")
        held_values.append(
            HeldPublicState(binding_kind, controlled_part, transform, uncertainties[0], uncertainties[1])
        )
    if offset != len(state_bytes):
        raise SchemaError("LENGTH", "public state parser did not consume exact bytes")
    digest = hashlib.sha256(state_bytes).hexdigest()
    if expected_sha256 is not None:
        if type(expected_sha256) is not str or not _SHA256_RE.fullmatch(expected_sha256):
            raise SchemaError("STATE_HASH", "expected state hash is malformed")
        if digest != expected_sha256:
            raise SchemaError("STATE_HASH", "public state hash mismatch")
    return PublicExecutionState(
        anchor,
        proprio,
        camera_mode,
        camera,
        (held_values[0], held_values[1]),
        state_bytes,
        digest,
    )


_RESOURCE_CONTEXT_KEYS = {
    "schema_id",
    "controller_target_enums",
    "constant_target_enums",
    "actuator_resource_ids",
    "compatible_wrist_target_pairs",
}
_RESOURCE_TARGET_JSON_KEYS = {
    "resource_id",
    "target_namespace",
    "target_selector",
    "target_key_hex",
}
_TARGET_KEY_LENGTHS = {
    TargetSelector.CONTROLLER_ENUM: 2,
    TargetSelector.LOCATOR_POINT: 4,
    TargetSelector.LOCATOR_AXIS: 6,
    TargetSelector.LOCATOR_FULL: 8,
    TargetSelector.ENTITY_LOCATOR: 13,
    TargetSelector.CONSTANT: 2,
}


def _resource_target_sort_key(
    target: tuple[int, TargetNamespace, TargetSelector, bytes],
) -> bytes:
    return (
        struct.pack("<H", target[0])
        + target[1].value.encode("ascii")
        + b"\x00"
        + target[2].value.encode("ascii")
        + b"\x00"
        + target[3]
    )


def _parse_wrist_target_json(
    value: object,
    *,
    name: str,
) -> tuple[int, TargetNamespace, TargetSelector, bytes]:
    obj = _exact_dict(value, _RESOURCE_TARGET_JSON_KEYS, prefix=name)
    resource_id = _strict_int(
        obj["resource_id"],
        "PROFILE_UNKNOWN_NOT_ADMITTED",
        f"{name}.resource_id",
        minimum=0,
        maximum=65535,
    )
    if type(obj["target_namespace"]) is not str:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", f"{name}.target_namespace must be string")
    if type(obj["target_selector"]) is not str:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", f"{name}.target_selector must be string")
    try:
        namespace = TargetNamespace(obj["target_namespace"])
        selector = TargetSelector(obj["target_selector"])
    except ValueError as exc:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", f"{name} has unknown target enum") from exc
    if namespace is not TargetNamespace.WRIST_POSE:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "wrist compatibility may contain only WRIST_POSE")
    key_hex = obj["target_key_hex"]
    if (
        type(key_hex) is not str
        or len(key_hex) % 2
        or re.fullmatch(r"[0-9a-f]*", key_hex) is None
    ):
        raise SchemaError(
            "PROFILE_UNKNOWN_NOT_ADMITTED",
            f"{name}.target_key_hex is not canonical",
        )
    key = bytes.fromhex(key_hex)
    if len(key) != _TARGET_KEY_LENGTHS[selector]:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", f"{name} target key length mismatch")
    return resource_id, namespace, selector, key


def validate_resource_validation_context(
    raw: bytes,
    *,
    expected_sha256: str,
) -> ResourceValidationContext:
    obj = _exact_dict(
        parse_strict_json_bytes(raw),
        _RESOURCE_CONTEXT_KEYS,
        prefix="resource validation context",
    )
    if obj["schema_id"] != "sm_fet_r1d_resource_validation_context_v1":
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "resource context schema mismatch")
    canonical = canonical_json_bytes(obj)
    digest = hashlib.sha256(canonical).hexdigest()
    _sha_field(expected_sha256, name="expected resource context hash", code="PROFILE_RECEIPT")
    if digest != expected_sha256:
        raise SchemaError("PROFILE_RECEIPT", "resource context hash mismatch")
    parsed_lists: list[tuple[int, ...]] = []
    for name in ("controller_target_enums", "constant_target_enums", "actuator_resource_ids"):
        values = obj[name]
        if type(values) is not list:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", f"{name} must be integer array")
        parsed = tuple(
            _strict_int(
                value,
                "PROFILE_UNKNOWN_NOT_ADMITTED",
                name,
                minimum=0,
                maximum=65535,
            )
            for value in values
        )
        if parsed != tuple(sorted(set(parsed))):
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", f"{name} must be sorted and unique")
        parsed_lists.append(parsed)
    raw_pairs = obj["compatible_wrist_target_pairs"]
    if type(raw_pairs) is not list:
        raise SchemaError(
            "PROFILE_UNKNOWN_NOT_ADMITTED",
            "compatible_wrist_target_pairs must be canonical array",
        )
    parsed_pairs: list[
        frozenset[tuple[int, TargetNamespace, TargetSelector, bytes]]
    ] = []
    pair_sort_keys: list[tuple[bytes, bytes]] = []
    for pair_index, raw_pair in enumerate(raw_pairs):
        if type(raw_pair) is not list or len(raw_pair) != 2:
            raise SchemaError(
                "PROFILE_UNKNOWN_NOT_ADMITTED",
                "wrist compatibility pair must have two targets",
            )
        left = _parse_wrist_target_json(raw_pair[0], name=f"wrist_pair[{pair_index}][0]")
        right = _parse_wrist_target_json(raw_pair[1], name=f"wrist_pair[{pair_index}][1]")
        left_key, right_key = _resource_target_sort_key(left), _resource_target_sort_key(right)
        if not left_key < right_key:
            raise SchemaError(
                "PROFILE_UNKNOWN_NOT_ADMITTED",
                "wrist pair targets must be distinct and bytewise sorted",
            )
        parsed_pairs.append(frozenset((left, right)))
        pair_sort_keys.append((left_key, right_key))
    if pair_sort_keys != sorted(set(pair_sort_keys)):
        raise SchemaError(
            "PROFILE_UNKNOWN_NOT_ADMITTED",
            "wrist compatibility pairs must be sorted and duplicate-free",
        )
    return ResourceValidationContext(
        parsed_lists[0],
        parsed_lists[1],
        parsed_lists[2],
        tuple(parsed_pairs),
        canonical,
        digest,
    )


def _validate_resource_context(
    context: ResourceValidationContext,
    profile_receipt: object,
) -> ResourceValidationContext:
    if type(context) is not ResourceValidationContext:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "resource context is not validated type")
    profile = _validated_profile_receipt_object(profile_receipt)
    if context.sha256 != profile["resource_validation_context_sha256"]:
        raise SchemaError("PROFILE_RECEIPT", "resource context is not bound by static profile")
    return validate_resource_validation_context(
        context.canonical_bytes,
        expected_sha256=profile["resource_validation_context_sha256"],
    )


def validate_resource_claim(
    claim: ResourceClaim,
    *,
    constraint: Constraint,
    context: ResourceValidationContext,
    verify: bool,
    profile_receipt: object,
) -> tuple[int, TargetNamespace, TargetSelector, bytes]:
    """Validate selector bytes against the exact canonical wire constraint."""

    if type(claim) is not ResourceClaim:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "claim must be ResourceClaim")
    _strict_bool(verify, "PROFILE_UNKNOWN_NOT_ADMITTED", "verify")
    context = _validate_resource_context(context, profile_receipt)
    _validate_constraint_fields(constraint, verify=verify)
    _strict_int(claim.resource_id, "PROFILE_UNKNOWN_NOT_ADMITTED", "resource_id", minimum=0, maximum=65535)
    if type(claim.claim_mode) is not ClaimMode:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "claim_mode is not frozen enum")
    if type(claim.target_namespace) is not TargetNamespace:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "target_namespace is not frozen enum")
    if type(claim.target_selector) is not TargetSelector:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "target_selector is not frozen enum")
    if type(claim.target_key) is not bytes:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "target_key must be finite bytes")
    if len(claim.target_key) != _TARGET_KEY_LENGTHS[claim.target_selector]:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "target_key length disagrees with selector")
    translation_rank = (constraint.locator.uv - 1) * 32 + constraint.locator.depth - 1
    full_locator_rank = locator_rank(constraint.locator, verify=verify)
    if claim.target_selector is TargetSelector.CONTROLLER_ENUM:
        controller_enum = int.from_bytes(claim.target_key, "little", signed=False)
        if controller_enum not in context.controller_target_enums:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "controller target enum is not frozen")
    elif claim.target_selector is TargetSelector.CONSTANT:
        constant = int.from_bytes(claim.target_key, "little", signed=False)
        if constant not in context.constant_target_enums:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "constant target enum is not frozen")
    elif claim.target_selector is TargetSelector.LOCATOR_POINT:
        if int.from_bytes(claim.target_key, "little", signed=False) != translation_rank:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "LOCATOR_POINT key differs from wire")
    elif claim.target_selector is TargetSelector.LOCATOR_AXIS:
        if constraint.locator.kind is LocatorKind.POINT:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "POINT has no canonical axis selector")
        point, axis = struct.unpack("<IH", claim.target_key)
        if point != translation_rank or axis != constraint.locator.axis or not 1 <= axis <= 512:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "LOCATOR_AXIS key differs from wire")
    elif claim.target_selector is TargetSelector.LOCATOR_FULL:
        if int.from_bytes(claim.target_key, "little", signed=False) != full_locator_rank:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "LOCATOR_FULL key differs from wire")
    else:
        expected = struct.pack(
            "<BBBBBQ",
            int(constraint.primary_role),
            int(constraint.part),
            int(constraint.secondary_role),
            int(constraint.relation),
            int(constraint.locator.kind),
            full_locator_rank,
        )
        if claim.target_key != expected:
            raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "ENTITY_LOCATOR key differs from wire")
    if claim.target_namespace is TargetNamespace.MONITORED_ENTITY and claim.claim_mode is not ClaimMode.READ_ONLY:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "monitored entity must be READ_ONLY")
    if claim.target_namespace is TargetNamespace.APERTURE_ACTUATION and claim.target_selector is not TargetSelector.CONTROLLER_ENUM:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "aperture target must use finite controller enum")
    return claim.resource_target_id


def _validate_resource_claim_set(
    constraint: Constraint,
    claims: Sequence[ResourceClaim],
    *,
    context: ResourceValidationContext,
    verify: bool,
    profile_receipt: object,
) -> None:
    if type(claims) not in (tuple, list) or not claims:
        raise SchemaError("PROFILE_UNKNOWN_NOT_ADMITTED", "constraint requires finite claim templates")
    resource_ids: list[int] = []
    targets: list[tuple[int, TargetNamespace, TargetSelector, bytes]] = []
    for claim in claims:
        targets.append(
            validate_resource_claim(
                claim,
                constraint=constraint,
                context=context,
                verify=verify,
                profile_receipt=profile_receipt,
            )
        )
        resource_ids.append(claim.resource_id)
    if len(set(resource_ids)) != len(resource_ids) or len(set(targets)) != len(targets):
        raise SchemaError(
            "PROFILE_UNKNOWN_NOT_ADMITTED",
            "one constraint has duplicate/ambiguous templates for a resource",
        )


def validate_physical_resource_claims(
    constraint: Constraint,
    claims: Sequence[ResourceClaim],
    *,
    context: ResourceValidationContext,
    verify: bool,
    profile_receipt: object,
) -> None:
    constraint_type = _constraint_type(constraint)
    _strict_bool(verify, "PROFILE_UNKNOWN_NOT_ADMITTED", "verify")
    if verify and constraint_type in (ConstraintType.LO, ConstraintType.RO):
        raise SchemaError("FUNCTION_ASSIGNMENT", "VERIFY physical claim cannot belong to OPERATE")
    context = _validate_resource_context(context, profile_receipt)
    _validate_resource_claim_set(
        constraint,
        claims,
        context=context,
        verify=verify,
        profile_receipt=profile_receipt,
    )
    if constraint_type is ConstraintType.ZM:
        for claim in claims:
            if (
                claim.claim_mode is not ClaimMode.READ_ONLY
                or claim.target_namespace is not TargetNamespace.MONITORED_ENTITY
            ):
                raise SchemaError(
                    "PROFILE_NO_ACTUATOR_CLAIM",
                    "MONITOR may claim only READ_ONLY monitored evidence",
                )
        return
    required_mode = (
        {ClaimMode.EXCLUSIVE_TARGET}
        if constraint.function is ConstraintFunction.OPERATE
        else {ClaimMode.EXCLUSIVE_TARGET, ClaimMode.HOLD_SAME_TARGET}
    )
    if not any(
        claim.claim_mode in required_mode and claim.resource_id in context.actuator_resource_ids
        for claim in claims
    ):
        raise SchemaError("PROFILE_NO_ACTUATOR_CLAIM", "physical O/S lacks required actuator claim")


def validate_resource_compatibility(
    left_constraint: Constraint,
    left: Sequence[ResourceClaim],
    right_constraint: Constraint,
    right: Sequence[ResourceClaim],
    *,
    context: ResourceValidationContext,
    verify: bool,
    profile_receipt: object,
) -> None:
    _strict_bool(verify, "PROFILE_UNKNOWN_NOT_ADMITTED", "verify")
    context = _validate_resource_context(context, profile_receipt)
    _validate_resource_claim_set(
        left_constraint,
        left,
        context=context,
        verify=verify,
        profile_receipt=profile_receipt,
    )
    _validate_resource_claim_set(
        right_constraint,
        right,
        context=context,
        verify=verify,
        profile_receipt=profile_receipt,
    )
    for first in left:
        for second in right:
            if (
                first.target_namespace is TargetNamespace.APERTURE_ACTUATION
                and second.target_namespace is TargetNamespace.APERTURE_ACTUATION
                and first.resource_target_id != second.resource_target_id
            ):
                raise SchemaError("PROFILE_RESOURCE_MUTEX", "distinct aperture target IDs are concurrent")
            if (
                first.target_namespace is TargetNamespace.WRIST_POSE
                and second.target_namespace is TargetNamespace.WRIST_POSE
                and first.resource_target_id != second.resource_target_id
                and frozenset((first.resource_target_id, second.resource_target_id))
                not in context.compatible_wrist_target_pairs
            ):
                raise SchemaError("PROFILE_RESOURCE_MUTEX", "distinct wrist targets require explicit profile compatibility")
            if (
                first.resource_id == second.resource_id
                and first.claim_mode is not ClaimMode.READ_ONLY
                and second.claim_mode is not ClaimMode.READ_ONLY
                and first.resource_target_id != second.resource_target_id
            ):
                raise SchemaError(
                    "PROFILE_RESOURCE_MUTEX",
                    "same finite resource has distinct action/hold targets",
                )


AXIS_VERTICAL_EPSILON_SQUARED: Final = float.fromhex("0x1.0000000000000p-80")
FRAME_ORTHONORMAL_TOLERANCE: Final = float.fromhex("0x1.0000000000000p-40")
RASTER_VERTICES: Final = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 0.5),
    (0.0, 0.5),
    (0.0, 1.0),
    (1.0, 1.0),
)
AXIS_CENTERS_LE_F64_SHA256: Final = "d9d9319054b954fceb759dd6c9c348f588813176b50466e54950691f0d14a5de"
FRAME_SURFACE_LE_F64_SHA256: Final = "0e0e05bad6118cd3b597f895a6ebf997842208f99a1f3580e834f1edd80073f9"
PATH_SURFACE_LE_F64_SHA256: Final = "191a9bdbdb35e030de8b915a441a1bdfce8d0bf691132d95c1e359c5c38eb41a"


def _strict_float(
    value: object,
    *,
    code: str,
    name: str,
    reject_negative_zero: bool = True,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SchemaError(code, f"{name} must be finite binary64")
    if reject_negative_zero and value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise SchemaError(code, f"{name} must not be negative zero")
    return value


def _vector3(
    value: object,
    *,
    code: str,
    name: str,
    reject_negative_zero: bool = True,
) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise SchemaError(code, f"{name} must be an exact length-3 tuple")
    return (
        _strict_float(
            value[0], code=code, name=f"{name}[0]", reject_negative_zero=reject_negative_zero
        ),
        _strict_float(
            value[1], code=code, name=f"{name}[1]", reject_negative_zero=reject_negative_zero
        ),
        _strict_float(
            value[2], code=code, name=f"{name}[2]", reject_negative_zero=reject_negative_zero
        ),
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: tuple[float, float, float], *, code: str, name: str) -> tuple[float, float, float]:
    length = _norm(value)
    if not math.isfinite(length) or length == 0.0:
        raise SchemaError(code, f"{name} must be finite and nonzero")
    return (value[0] / length, value[1] / length, value[2] / length)


@lru_cache(maxsize=1)
def generated_axis_centers() -> tuple[tuple[float, float, float], ...]:
    centers: list[tuple[float, float, float]] = []
    for elevation_index in range(16):
        elevation = -math.pi / 2.0 + math.pi * (elevation_index + 0.5) / 16.0
        for azimuth_index in range(32):
            azimuth = -math.pi + 2.0 * math.pi * (azimuth_index + 0.5) / 32.0
            centers.append(
                (
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                )
            )
    return tuple(centers)


def quantize_normalized_uv(u: float, v: float) -> int:
    u = _strict_float(u, code="METRIC_UNKNOWN", name="u")
    v = _strict_float(v, code="METRIC_UNKNOWN", name="v")
    if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        raise SchemaError("METRIC_UNKNOWN", "normalized UV is outside [0,1]")
    column = 63 if u == 1.0 else math.floor(u * 64.0)
    row = 63 if v == 1.0 else math.floor(v * 64.0)
    return 1 + 64 * row + column


def quantize_pixel_center(pixel_x: float, pixel_y: float, *, width: int, height: int) -> int:
    pixel_x = _strict_float(pixel_x, code="METRIC_UNKNOWN", name="pixel_x")
    pixel_y = _strict_float(pixel_y, code="METRIC_UNKNOWN", name="pixel_y")
    _strict_int(width, "CODEBOOK_W_H_MISSING_OR_MISMATCH", "width", minimum=1, maximum=(1 << 32) - 1)
    _strict_int(height, "CODEBOOK_W_H_MISSING_OR_MISMATCH", "height", minimum=1, maximum=(1 << 32) - 1)
    if not 0.0 <= pixel_x < width or not 0.0 <= pixel_y < height:
        raise SchemaError("METRIC_UNKNOWN", "pixel center is outside preprocessed raster")
    return quantize_normalized_uv((pixel_x + 0.5) / width, (pixel_y + 0.5) / height)


def quantize_metric_cell(value: float, edges: tuple[float, ...], *, expected_bins: int) -> int:
    value = _strict_float(value, code="METRIC_UNKNOWN", name="metric value")
    _strict_int(expected_bins, "CODEBOOK_TABLES", "expected_bins", minimum=1, maximum=65535)
    if type(edges) is not tuple or len(edges) != expected_bins + 1:
        raise SchemaError("CODEBOOK_TABLES", "metric edges have wrong canonical shape")
    parsed = tuple(
        _strict_float(edge, code="CODEBOOK_TABLES", name=f"edge[{index}]")
        for index, edge in enumerate(edges)
    )
    if not _strictly_increasing(parsed):
        raise SchemaError("CODEBOOK_TABLES", "metric edges are not strictly increasing")
    if value < parsed[0] or value > parsed[-1]:
        raise SchemaError("METRIC_UNKNOWN", "metric value is outside admitted range")
    if value == parsed[-1]:
        return expected_bins
    return bisect_right(parsed, value)


def quantize_depth(depth_camera_z_m: float, depth_edges: tuple[float, ...]) -> int:
    if type(depth_edges) is not tuple or any(
        type(value) is not float or not math.isfinite(value) or value < 0.0
        for value in depth_edges
    ):
        raise SchemaError("CODEBOOK_TABLES", "depth edges must be nonnegative binary64")
    return quantize_metric_cell(depth_camera_z_m, depth_edges, expected_bins=32)


def quantize_extent(extent_m: float, extent_edges: tuple[float, ...]) -> int:
    if type(extent_edges) is not tuple or any(
        type(value) is not float or not math.isfinite(value) or value <= 0.0
        for value in extent_edges
    ):
        raise SchemaError("CODEBOOK_TABLES", "extent edges must be positive binary64")
    return quantize_metric_cell(extent_m, extent_edges, expected_bins=8)


def decode_translation_center(
    uv_code: int,
    depth_code: int,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_centers: tuple[float, ...],
) -> tuple[float, float, float]:
    _strict_int(uv_code, "FIELD_RANGE", "uv_code", minimum=1, maximum=4096)
    _strict_int(depth_code, "FIELD_RANGE", "depth_code", minimum=1, maximum=32)
    _strict_int(width, "CODEBOOK_W_H_MISSING_OR_MISMATCH", "width", minimum=1, maximum=(1 << 32) - 1)
    _strict_int(height, "CODEBOOK_W_H_MISSING_OR_MISMATCH", "height", minimum=1, maximum=(1 << 32) - 1)
    fx = _strict_float(fx, code="CODEBOOK_TABLES", name="fx")
    fy = _strict_float(fy, code="CODEBOOK_TABLES", name="fy")
    cx = _strict_float(cx, code="CODEBOOK_TABLES", name="cx")
    cy = _strict_float(cy, code="CODEBOOK_TABLES", name="cy")
    if fx <= 0.0 or fy <= 0.0 or type(depth_centers) is not tuple or len(depth_centers) != 32:
        raise SchemaError("CODEBOOK_TABLES", "intrinsics/depth centers are invalid")
    centers = tuple(
        _strict_float(value, code="CODEBOOK_TABLES", name=f"depth_center[{index}]")
        for index, value in enumerate(depth_centers)
    )
    uv_index = uv_code - 1
    row, column = divmod(uv_index, 64)
    u_center = (column + 0.5) / 64.0
    v_center = (row + 0.5) / 64.0
    pixel_x = u_center * width - 0.5
    pixel_y = v_center * height - 0.5
    z_center = centers[depth_code - 1]
    return (
        (pixel_x - cx) * z_center / fx,
        (pixel_y - cy) * z_center / fy,
        z_center,
    )


def quantize_axis(direction: tuple[float, float, float]) -> int:
    unit = _normalize(
        _vector3(direction, code="METRIC_UNKNOWN", name="axis"),
        code="METRIC_UNKNOWN",
        name="axis",
    )
    # Python binary64 operations are sequenced explicitly; no fused multiply-add.
    x2 = unit[0] * unit[0]
    y2 = unit[1] * unit[1]
    rho2 = x2 + y2
    if rho2 <= AXIS_VERTICAL_EPSILON_SQUARED:
        azimuth_index = 0
        elevation_index = 15 if unit[2] > 0.0 else 0
    else:
        azimuth = math.atan2(unit[1], unit[0])
        if azimuth == math.pi:
            azimuth = -math.pi
        elevation = math.asin(max(-1.0, min(1.0, unit[2])))
        azimuth_edges = tuple(-math.pi + 2.0 * math.pi * index / 32.0 for index in range(33))
        elevation_edges = tuple(-math.pi / 2.0 + math.pi * index / 16.0 for index in range(17))
        azimuth_index = min(31, bisect_right(azimuth_edges, azimuth) - 1)
        elevation_index = min(15, bisect_right(elevation_edges, elevation) - 1)
    return 1 + 32 * elevation_index + azimuth_index


def decode_axis_center(axis_code: int) -> tuple[float, float, float]:
    _strict_int(axis_code, "FIELD_RANGE", "axis_code", minimum=1, maximum=512)
    return generated_axis_centers()[axis_code - 1]


@lru_cache(maxsize=512, typed=True)
def zero_roll_basis(axis_code: int) -> FrameBasis:
    z_axis = decode_axis_center(axis_code)
    seeds = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    seed_index = min(range(3), key=lambda index: (abs(_dot(seeds[index], z_axis)), index))
    x_axis = _normalize(
        _cross(seeds[seed_index], z_axis),
        code="FRAME_INVALID",
        name="zero-roll x",
    )
    y_axis = _cross(z_axis, x_axis)
    frame = FrameBasis(x_axis, y_axis, z_axis)
    validate_frame_basis(frame)
    return frame


def validate_frame_basis(frame: FrameBasis) -> None:
    if type(frame) is not FrameBasis:
        raise SchemaError("FRAME_INVALID", "frame must be FrameBasis")
    x_axis = _vector3(
        frame.x, code="FRAME_INVALID", name="frame.x", reject_negative_zero=False
    )
    y_axis = _vector3(
        frame.y, code="FRAME_INVALID", name="frame.y", reject_negative_zero=False
    )
    z_axis = _vector3(
        frame.z, code="FRAME_INVALID", name="frame.z", reject_negative_zero=False
    )
    tolerance = FRAME_ORTHONORMAL_TOLERANCE
    if any(abs(_norm(axis) - 1.0) > tolerance for axis in (x_axis, y_axis, z_axis)):
        raise SchemaError("FRAME_INVALID", "frame axis is not unit")
    if any(abs(value) > tolerance for value in (_dot(x_axis, y_axis), _dot(x_axis, z_axis), _dot(y_axis, z_axis))):
        raise SchemaError("FRAME_INVALID", "frame axes are not orthogonal")
    if _dot(_cross(x_axis, y_axis), z_axis) < 1.0 - tolerance:
        raise SchemaError("FRAME_INVALID", "frame is not right-handed")


@lru_cache(maxsize=512 * 32, typed=True)
def decode_frame(axis_code: int, roll_code: int) -> FrameBasis:
    _strict_int(roll_code, "FIELD_RANGE", "roll_code", minimum=1, maximum=32)
    base = zero_roll_basis(axis_code)
    theta = 2.0 * math.pi * (roll_code - 1) / 32.0
    x_axis = tuple(
        math.cos(theta) * base.x[index] + math.sin(theta) * base.y[index]
        for index in range(3)
    )
    y_axis = _cross(base.z, x_axis)
    frame = FrameBasis(x_axis, y_axis, base.z)
    validate_frame_basis(frame)
    return frame


def quantize_roll(axis_code: int, x_direction: tuple[float, float, float]) -> int:
    base = zero_roll_basis(axis_code)
    raw_direction = _vector3(
        x_direction,
        code="METRIC_UNKNOWN",
        name="roll x direction",
    )
    direction = _normalize(
        raw_direction,
        code="METRIC_UNKNOWN",
        name="roll x direction",
    )
    # The 32 materialized decoded directions are constructive exact ties.  This
    # finite equality check has zero width: adjacent representable inputs are
    # never absorbed into the edge as they were by the removed ULP band.
    for roll_code in range(1, 33):
        golden_direction = decode_frame(axis_code, roll_code).x
        if raw_direction == golden_direction or direction == golden_direction:
            return roll_code
    coordinate_x = _dot(direction, base.x)
    coordinate_y = _dot(direction, base.y)
    if coordinate_x == 0.0 and coordinate_y == 0.0:
        raise SchemaError("METRIC_UNKNOWN", "roll direction is parallel to axis")
    theta = math.atan2(coordinate_y, coordinate_x)
    if theta < 0.0:
        theta += 2.0 * math.pi
    if theta == 2.0 * math.pi:
        theta = 0.0
    edges = tuple(2.0 * math.pi * index / 32.0 for index in range(33))
    return 1 + min(31, bisect_right(edges, theta) - 1)


def _frame_apply(frame: FrameBasis, local: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        frame.x[index] * local[0] + frame.y[index] * local[1] + frame.z[index] * local[2]
        for index in range(3)
    )


def sample_path_pose(
    primitive: PathPrimitive,
    parameter: float,
    extent_m: float,
    *,
    anchor: tuple[float, float, float],
    axis_code: int,
    roll_code: int,
) -> PathPose:
    if type(primitive) is not PathPrimitive:
        raise SchemaError("PATH_INVALID", "primitive must be PathPrimitive")
    parameter = _strict_float(parameter, code="PATH_INVALID", name="parameter")
    extent_m = _strict_float(extent_m, code="PATH_INVALID", name="extent_m")
    anchor = _vector3(anchor, code="PATH_INVALID", name="anchor")
    if not 0.0 <= parameter <= 1.0 or extent_m <= 0.0:
        raise SchemaError("PATH_INVALID", "path parameter/extent is outside range")
    if primitive is PathPrimitive.LINE:
        gamma = (parameter, 0.0)
        tangent = (1.0, 0.0)
    elif primitive in (PathPrimitive.ARC_LEFT, PathPrimitive.ARC_RIGHT):
        theta = math.pi * parameter / 2.0
        sign = 1.0 if primitive is PathPrimitive.ARC_LEFT else -1.0
        gamma = (math.sin(theta), sign * (1.0 - math.cos(theta)))
        tangent = (math.cos(theta), sign * math.sin(theta))
    else:
        cumulative = (0.0, 1.0, 1.5, 2.5, 3.0, 4.0)
        distance = 4.0 * parameter
        segment = 4 if parameter == 1.0 else bisect_right(cumulative, distance) - 1
        start, end = RASTER_VERTICES[segment], RASTER_VERTICES[segment + 1]
        segment_length = cumulative[segment + 1] - cumulative[segment]
        local = (distance - cumulative[segment]) / segment_length
        gamma = (start[0] + local * (end[0] - start[0]), start[1] + local * (end[1] - start[1]))
        tangent = ((end[0] - start[0]) / segment_length, (end[1] - start[1]) / segment_length)
    anchor_frame = decode_frame(axis_code, roll_code)
    displacement = _frame_apply(anchor_frame, (extent_m * gamma[0], extent_m * gamma[1], 0.0))
    position = tuple(anchor[index] + displacement[index] for index in range(3))
    x_axis = _normalize(
        _frame_apply(anchor_frame, (tangent[0], tangent[1], 0.0)),
        code="PATH_INVALID",
        name="path tangent",
    )
    y_axis = _cross(anchor_frame.z, x_axis)
    frame = FrameBasis(x_axis, y_axis, anchor_frame.z)
    validate_frame_basis(frame)
    return PathPose(parameter, position, frame)


def verify_frozen_geometry_golden_surfaces() -> dict[str, str]:
    """Recompute independent literal digests for axis, frame, and path surfaces."""

    axes = generated_axis_centers()
    axis_bytes = struct.pack("<1536d", *(value for axis in axes for value in axis))
    frame_values: list[float] = []
    for axis_code in range(1, 513):
        for roll_code in range(1, 33):
            frame = decode_frame(axis_code, roll_code)
            frame_values.extend((*frame.x, *frame.y, *frame.z))
    frame_bytes = struct.pack(f"<{len(frame_values)}d", *frame_values)
    path_values: list[float] = []
    for primitive in PathPrimitive:
        for index in range(65):
            pose = sample_path_pose(
                primitive,
                index / 64.0,
                0.25,
                anchor=(0.0, 0.0, 0.0),
                axis_code=257,
                roll_code=1,
            )
            path_values.extend(
                (pose.parameter, *pose.position, *pose.frame.x, *pose.frame.y, *pose.frame.z)
            )
    path_bytes = struct.pack(f"<{len(path_values)}d", *path_values)
    actual = {
        "axis_centers_le_f64_sha256": hashlib.sha256(axis_bytes).hexdigest(),
        "frame_surface_le_f64_sha256": hashlib.sha256(frame_bytes).hexdigest(),
        "path_surface_le_f64_sha256": hashlib.sha256(path_bytes).hexdigest(),
    }
    expected = {
        "axis_centers_le_f64_sha256": AXIS_CENTERS_LE_F64_SHA256,
        "frame_surface_le_f64_sha256": FRAME_SURFACE_LE_F64_SHA256,
        "path_surface_le_f64_sha256": PATH_SURFACE_LE_F64_SHA256,
    }
    if actual != expected:
        raise SchemaError("GOLDEN_SURFACE", f"geometry golden digest mismatch: {actual!r}")
    boundary = (
        quantize_normalized_uv(0.0, 0.0),
        quantize_normalized_uv(1.0 / 64.0, 0.0),
        quantize_normalized_uv(1.0, 1.0),
        quantize_axis((0.0, 0.0, -1.0)),
        quantize_axis((0.0, 0.0, 1.0)),
    )
    if boundary != (1, 2, 4096, 1, 481):
        raise SchemaError("GOLDEN_SURFACE", f"quantizer boundary mismatch: {boundary!r}")
    return actual


def project_transform(kind: LocatorKind, transform: RigidTransform) -> GoalProjection:
    if type(kind) is not LocatorKind or kind is LocatorKind.PATH:
        raise SchemaError("GOAL_MANIFOLD", "projection kind must be POINT/AXIS/FRAME")
    if type(transform) is not RigidTransform:
        raise SchemaError("GOAL_MANIFOLD", "transform must be RigidTransform")
    position = _vector3(transform.translation, code="GOAL_MANIFOLD", name="translation")
    validate_frame_basis(transform.rotation)
    if kind is LocatorKind.POINT:
        return GoalProjection(kind, position)
    if kind is LocatorKind.AXIS:
        return GoalProjection(kind, position, axis=transform.rotation.z)
    return GoalProjection(kind, position, axis=transform.rotation.z, frame=transform.rotation)


def validate_verify_hold_existing(
    goal: GoalProjection,
    current_effector: RigidTransform,
    *,
    validation_policy: PublicStateValidationPolicy,
    profile_receipt: object,
) -> None:
    """Generic VERIFY HOLD_EXISTING oracle over POINT/AXIS/FRAME manifolds."""

    policy = _resolve_bound_validation_policy(validation_policy, profile_receipt)
    if type(goal) is not GoalProjection or type(current_effector) is not RigidTransform:
        raise SchemaError("VERIFY_HOLD_NOT_SATISFIED", "goal/current types are invalid")
    if type(goal.kind) is not LocatorKind or goal.kind is LocatorKind.PATH:
        raise SchemaError("GOAL_MANIFOLD", "VERIFY goal kind must be POINT/AXIS/FRAME")
    if goal.kind is LocatorKind.POINT and (goal.axis is not None or goal.frame is not None):
        raise SchemaError("GOAL_MANIFOLD", "POINT goal must leave orientation free")
    if goal.kind is LocatorKind.AXIS and (goal.axis is None or goal.frame is not None):
        raise SchemaError("GOAL_MANIFOLD", "AXIS goal must specify axis and leave roll free")
    if goal.kind is LocatorKind.FRAME:
        if type(goal.frame) is not FrameBasis or goal.axis != goal.frame.z:
            raise SchemaError("GOAL_MANIFOLD", "FRAME goal must specify one consistent full frame")
    current = project_transform(goal.kind, current_effector)
    target_position = _vector3(goal.position, code="GOAL_MANIFOLD", name="goal.position")
    position_error = _norm(tuple(current.position[index] - target_position[index] for index in range(3)))
    if position_error > policy.verify_position_tolerance_m:
        raise SchemaError("VERIFY_HOLD_NOT_SATISFIED", "effector origin is outside tolerance")
    if goal.kind is LocatorKind.AXIS:
        target_axis = _normalize(
            _vector3(goal.axis, code="GOAL_MANIFOLD", name="goal.axis"),
            code="GOAL_MANIFOLD",
            name="goal.axis",
        )
        axis_error = math.acos(max(-1.0, min(1.0, _dot(current.axis, target_axis))))  # type: ignore[arg-type]
        if axis_error > policy.verify_axis_tolerance_rad:
            raise SchemaError("VERIFY_HOLD_NOT_SATISFIED", "effector axis is outside tolerance")
    elif goal.kind is LocatorKind.FRAME:
        if type(goal.frame) is not FrameBasis:
            raise SchemaError("GOAL_MANIFOLD", "FRAME goal requires full frame")
        validate_frame_basis(goal.frame)
        trace = sum(
            _dot(current_axis, target_axis)
            for current_axis, target_axis in zip(
                (current.frame.x, current.frame.y, current.frame.z),  # type: ignore[union-attr]
                (goal.frame.x, goal.frame.y, goal.frame.z),
                strict=True,
            )
        )
        rotation_error = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
        if rotation_error > policy.verify_frame_tolerance_rad:
            raise SchemaError("VERIFY_HOLD_NOT_SATISFIED", "effector frame is outside tolerance")


_METRIC_MANIFEST_KEYS = {
    "schema_id",
    "generator_version",
    "locator_width_px",
    "locator_height_px",
    "source_calibration_sha256",
    "preprocess_id",
    "camera_convention",
    "pixel_center_rule",
    "uv_grid",
    "depth_bins",
    "axis_bins",
    "roll_bins",
    "path_primitives",
    "extent_bins",
    "edge_tie_rule",
    "float_policy",
    "angle_quantizer",
    "depth_unit",
    "extent_unit",
    "axis_generator",
    "axis_vertical_epsilon_squared_hex",
    "roll_generator",
    "path_generator",
    "table_layout",
}
_METRIC_CONSTANTS = {
    "schema_id": METRIC_MANIFEST_SCHEMA_ID,
    "generator_version": "sm_fet_r1d_codebook_generator_v1",
    "camera_convention": "OPENCV_RIGHT_DOWN_FORWARD",
    "pixel_center_rule": "HALF_PIXEL_NORMALIZED_V1",
    "depth_bins": 32,
    "axis_bins": 512,
    "roll_bins": 32,
    "path_primitives": 4,
    "extent_bins": 8,
    "edge_tie_rule": "INTERNAL_TO_HIGH_LAST_RIGHT_CLOSED_V1",
    "float_policy": "IEEE754_BINARY64_RNE_NO_FMA_V1",
    "angle_quantizer": "CORRECTLY_ROUNDED_BINARY64_ATAN2_ASIN_V1",
    "depth_unit": "METER_CAMERA_Z",
    "extent_unit": "METER",
    "axis_generator": "r1d_axis_az32_el16_equal_angle_v1",
    "axis_vertical_epsilon_squared_hex": "0x1p-80",
    "roll_generator": "r1d_roll32_seed_min_absdot_v1",
    "path_generator": "r1d_line_arc90_raster3_v1",
    "table_layout": "sm_fet_r1d_tables_le_f64_v1",
}
_TABLE_DOUBLE_COUNT = 1718
_TABLE_BYTES = _TABLE_DOUBLE_COUNT * 8


def validate_metric_manifest(manifest: object) -> bytes:
    obj = _exact_dict(manifest, _METRIC_MANIFEST_KEYS, prefix="metric manifest")
    for key, expected in _METRIC_CONSTANTS.items():
        if type(obj[key]) is not type(expected) or obj[key] != expected:
            raise SchemaError("CODEBOOK_MANIFEST", f"manifest {key} mismatch")
    for key in ("locator_width_px", "locator_height_px"):
        _strict_int(obj[key], "CODEBOOK_W_H_MISSING_OR_MISMATCH", key, minimum=1, maximum=(1 << 32) - 1)
    if type(obj["uv_grid"]) is not list or obj["uv_grid"] != [64, 64] or any(
        type(value) is not int for value in obj["uv_grid"]
    ):
        raise SchemaError("CODEBOOK_MANIFEST", "uv_grid must be exact [64,64]")
    if type(obj["source_calibration_sha256"]) is not str or not _SHA256_RE.fullmatch(
        obj["source_calibration_sha256"]
    ):
        raise SchemaError("CODEBOOK_MANIFEST", "calibration hash is malformed")
    _identifier(obj["preprocess_id"], name="preprocess_id", code="CODEBOOK_MANIFEST")
    return canonical_json_bytes(obj)


def _double_values(raw: bytes) -> tuple[float, ...]:
    if type(raw) is not bytes or len(raw) != _TABLE_BYTES:
        raise SchemaError("CODEBOOK_TABLES", f"tables.bin must be exactly {_TABLE_BYTES} bytes")
    values = struct.unpack(f"<{_TABLE_DOUBLE_COUNT}d", raw)
    bits = struct.unpack(f"<{_TABLE_DOUBLE_COUNT}Q", raw)
    for index, (value, word) in enumerate(zip(values, bits, strict=True)):
        if not math.isfinite(value):
            raise SchemaError("CODEBOOK_TABLES", f"tables[{index}] is NaN/Inf")
        if word == 0x8000000000000000:
            raise SchemaError("CODEBOOK_TABLES", f"tables[{index}] is negative zero")
    return values


def _strictly_increasing(values: Sequence[float], *, positive: bool = False) -> bool:
    return all((not positive or value > 0.0) for value in values) and all(
        left < right for left, right in zip(values, values[1:])
    )


def validate_metric_codebook(
    manifest: object,
    tables_bin: bytes,
    *,
    expected_codebook_sha256: str,
) -> str:
    """Validate a materialized metric artifact and recompute its checksum."""

    manifest_object = parse_strict_json_bytes(manifest) if type(manifest) is bytes else manifest
    manifest_bytes = validate_metric_manifest(manifest_object)
    if type(expected_codebook_sha256) is not str or not _SHA256_RE.fullmatch(expected_codebook_sha256):
        raise SchemaError("CODEBOOK_HASH", "expected codebook hash is malformed or UNDEFINED")
    values = _double_values(tables_bin)
    cursor = 0

    def take(count: int) -> tuple[float, ...]:
        nonlocal cursor
        result = values[cursor : cursor + count]
        cursor += count
        return result

    fx, fy, cx, cy = take(4)
    if fx <= 0.0 or fy <= 0.0:
        raise SchemaError("CODEBOOK_W_H_MISSING_OR_MISMATCH", "focal lengths must be positive")
    width = manifest_object["locator_width_px"]  # type: ignore[index]
    height = manifest_object["locator_height_px"]  # type: ignore[index]
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        raise SchemaError("CODEBOOK_W_H_MISSING_OR_MISMATCH", "intrinsics are outside manifest raster")
    z_edges, z_centers = take(33), take(32)
    azimuth_edges, elevation_edges = take(33), take(17)
    axes_flat = take(512 * 3)
    roll_edges = take(33)
    extent_edges, extent_centers = take(9), take(8)
    path_pi_over_2 = take(1)[0]
    raster = take(12)
    if cursor != _TABLE_DOUBLE_COUNT:
        raise SchemaError("CODEBOOK_TABLES", "table layout count mismatch")
    if not _strictly_increasing(z_edges) or z_edges[0] < 0.0:
        raise SchemaError("CODEBOOK_TABLES", "depth edges must be finite strictly increasing")
    if not _strictly_increasing(extent_edges, positive=True):
        raise SchemaError("CODEBOOK_TABLES", "extent edges must be positive strictly increasing")
    for centers, edges, name in (
        (z_centers, z_edges, "depth"),
        (extent_centers, extent_edges, "extent"),
    ):
        if any(
            not (
                edges[index] <= center < edges[index + 1]
                if index < len(centers) - 1
                else edges[index] <= center <= edges[index + 1]
            )
            for index, center in enumerate(centers)
        ):
            raise SchemaError("CODEBOOK_TABLES", f"{name} center lies outside its cell")
    if not _strictly_increasing(azimuth_edges) or not _strictly_increasing(elevation_edges):
        raise SchemaError("CODEBOOK_TABLES", "axis angle edges must be strictly increasing")
    if not _strictly_increasing(roll_edges):
        raise SchemaError("CODEBOOK_TABLES", "roll edges must be strictly increasing")
    expected_azimuth_edges = tuple(-math.pi + 2.0 * math.pi * index / 32.0 for index in range(33))
    expected_elevation_edges = tuple(-math.pi / 2.0 + math.pi * index / 16.0 for index in range(17))
    expected_roll_edges = tuple(2.0 * math.pi * index / 32.0 for index in range(33))
    if azimuth_edges != expected_azimuth_edges or elevation_edges != expected_elevation_edges:
        raise SchemaError("CODEBOOK_TABLES", "axis edges differ from frozen generator")
    if roll_edges != expected_roll_edges:
        raise SchemaError("CODEBOOK_TABLES", "roll edges differ from frozen generator")
    axes = tuple(tuple(axes_flat[index : index + 3]) for index in range(0, len(axes_flat), 3))
    if any(
        abs(math.sqrt(sum(component * component for component in axis)) - 1.0)
        > AXIS_TABLE_UNIT_TOLERANCE
        for axis in axes
    ):
        raise SchemaError("CODEBOOK_TABLES", "axis center is outside unit tolerance")
    if len({struct.pack("<3d", *axis) for axis in axes}) != 512:
        raise SchemaError("CODEBOOK_TABLES", "axis centers are not bytewise distinct")
    expected_axes = []
    for elevation_index in range(16):
        elevation = -math.pi / 2.0 + math.pi * (elevation_index + 0.5) / 16.0
        for azimuth_index in range(32):
            azimuth = -math.pi + 2.0 * math.pi * (azimuth_index + 0.5) / 32.0
            expected_axes.append(
                (
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                )
            )
    if axes != tuple(expected_axes):
        raise SchemaError("CODEBOOK_TABLES", "axis centers differ from frozen generator")
    if path_pi_over_2 != math.pi / 2.0:
        raise SchemaError("CODEBOOK_TABLES", "path pi/2 constant mismatch")
    expected_raster = (0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 1.0, 1.0, 1.0)
    if raster != expected_raster:
        raise SchemaError("CODEBOOK_TABLES", "RASTER_3 vertices mismatch")
    digest = hashlib.sha256(manifest_bytes + b"\x00" + tables_bin).hexdigest()
    if digest != expected_codebook_sha256:
        raise SchemaError("CODEBOOK_HASH", "metric codebook checksum mismatch")
    return digest


def require_defined_codebook(
    *,
    receipt: object,
    manifest: object,
    tables_bin: bytes,
    profile_receipt: object,
    authorized_codebook_receipt_sha256: str,
) -> str:
    """Return a codebook hash only after full artifact/profile authorization."""

    try:
        authorized = _sha_field(
            authorized_codebook_receipt_sha256,
            name="authorized codebook receipt",
            code="CODEBOOK_UNDEFINED",
        )
        receipt_hash = validate_metric_codebook_receipt(
            receipt,
            manifest=manifest,
            tables_bin=tables_bin,
        )
        if receipt_hash != authorized:
            raise SchemaError("CODEBOOK_UNDEFINED", "codebook receipt is not authorized")
        codebook = _validate_self_hashing_receipt(
            receipt,
            keys=_CODEBOOK_RECEIPT_KEYS,
            schema_id="metric_codebook_receipt_v1",
            code="CODEBOOK_RECEIPT",
        )
        profile = _validated_profile_receipt_object(profile_receipt)
        if codebook["codebook_sha256"] != profile["metric_codebook_sha256"]:
            raise SchemaError("CODEBOOK_UNDEFINED", "profile is bound to a different codebook")
        return codebook["codebook_sha256"]
    except SchemaError as exc:
        if exc.code == "CODEBOOK_UNDEFINED":
            raise
        raise SchemaError("CODEBOOK_UNDEFINED", "no verified authorized metric artifact") from exc


_PROFILE_RECEIPT_KEYS = {
    "schema_id",
    "profile_id",
    "hardware_family",
    "task_stratum_id",
    "hardware_manifest_sha256",
    "metric_codebook_sha256",
    "profile_payload_sha256",
    "public_state_validation_policy_sha256",
    "resource_validation_context_sha256",
    "receipt_sha256",
}


def _sha_field(value: object, *, name: str, code: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise SchemaError(code, f"{name} must be 64 lowercase hex chars")
    return value


def _validate_self_hashing_receipt(
    receipt: object,
    *,
    keys: set[str],
    schema_id: str,
    code: str,
) -> dict[str, Any]:
    obj = _exact_dict(receipt, keys, prefix=schema_id)
    if obj["schema_id"] != schema_id:
        raise SchemaError(code, "receipt schema_id mismatch")
    supplied = _sha_field(obj["receipt_sha256"], name="receipt_sha256", code=code)
    payload = {key: obj[key] for key in obj if key != "receipt_sha256"}
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != supplied:
        raise SchemaError(code, "receipt self-hash mismatch")
    return obj


def _validated_profile_receipt_object(receipt: object) -> dict[str, Any]:
    obj = _validate_self_hashing_receipt(
        receipt,
        keys=_PROFILE_RECEIPT_KEYS,
        schema_id="static_capability_profile_receipt_v1",
        code="PROFILE_RECEIPT",
    )
    _identifier(obj["profile_id"], name="profile_id", code="PROFILE_RECEIPT")
    if obj["hardware_family"] != "unitree_g1_left_rh56dftp_right_omnipicker":
        raise SchemaError("PROFILE_RECEIPT", "hardware family mismatch")
    _identifier(obj["task_stratum_id"], name="task_stratum_id", code="PROFILE_RECEIPT")
    for field in (
        "hardware_manifest_sha256",
        "metric_codebook_sha256",
        "profile_payload_sha256",
        "public_state_validation_policy_sha256",
        "resource_validation_context_sha256",
    ):
        _sha_field(obj[field], name=field, code="PROFILE_RECEIPT")
    return obj


def validate_static_capability_profile_receipt(receipt: object) -> str:
    obj = _validated_profile_receipt_object(receipt)
    return obj["receipt_sha256"]


_CODEBOOK_RECEIPT_KEYS = {
    "schema_id",
    "codebook_sha256",
    "source_calibration_sha256",
    "manifest_sha256",
    "tables_sha256",
    "receipt_sha256",
}


def validate_metric_codebook_receipt(
    receipt: object,
    *,
    manifest: object,
    tables_bin: bytes,
) -> str:
    obj = _validate_self_hashing_receipt(
        receipt,
        keys=_CODEBOOK_RECEIPT_KEYS,
        schema_id="metric_codebook_receipt_v1",
        code="CODEBOOK_RECEIPT",
    )
    for field in (
        "codebook_sha256",
        "source_calibration_sha256",
        "manifest_sha256",
        "tables_sha256",
    ):
        _sha_field(obj[field], name=field, code="CODEBOOK_RECEIPT")
    manifest_object = parse_strict_json_bytes(manifest) if type(manifest) is bytes else manifest
    manifest_bytes = validate_metric_manifest(manifest_object)
    if hashlib.sha256(manifest_bytes).hexdigest() != obj["manifest_sha256"]:
        raise SchemaError("CODEBOOK_RECEIPT", "manifest hash mismatch")
    if hashlib.sha256(tables_bin).hexdigest() != obj["tables_sha256"]:
        raise SchemaError("CODEBOOK_RECEIPT", "tables hash mismatch")
    if manifest_object["source_calibration_sha256"] != obj["source_calibration_sha256"]:  # type: ignore[index]
        raise SchemaError("CODEBOOK_RECEIPT", "calibration hash layer mismatch")
    validate_metric_codebook(manifest, tables_bin, expected_codebook_sha256=obj["codebook_sha256"])
    return obj["receipt_sha256"]


_HELD_RECEIPT_KEYS = {
    "schema_id",
    "public_state_sha256",
    "source_id",
    "source_manifest_sha256",
    "calibration_sha256",
    "anchor_timestamp_ns",
    "freshness_deadline_ns",
    "receipt_sha256",
}


def validate_held_public_state_receipt(
    receipt: object,
    *,
    public_state: PublicExecutionState,
    now_ns: int,
) -> str:
    _strict_int(now_ns, "STATE_STALE", "now_ns", minimum=-(1 << 63), maximum=(1 << 63) - 1)
    if type(public_state) is not PublicExecutionState:
        raise SchemaError("HELD_RECEIPT", "public_state must be validated PublicExecutionState")
    obj = _validate_self_hashing_receipt(
        receipt,
        keys=_HELD_RECEIPT_KEYS,
        schema_id="held_public_state_receipt_v1",
        code="HELD_RECEIPT",
    )
    if obj["public_state_sha256"] != public_state.sha256:
        raise SchemaError("STATE_HASH", "held receipt does not bind public-state bytes")
    _identifier(obj["source_id"], name="source_id", code="HELD_RECEIPT")
    for field in ("source_manifest_sha256", "calibration_sha256"):
        _sha_field(obj[field], name=field, code="HELD_RECEIPT")
    anchor = _strict_int(
        obj["anchor_timestamp_ns"],
        "HELD_RECEIPT",
        "anchor_timestamp_ns",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    deadline = _strict_int(
        obj["freshness_deadline_ns"],
        "HELD_RECEIPT",
        "freshness_deadline_ns",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    if anchor != public_state.anchor_timestamp_ns or now_ns > deadline or deadline < anchor:
        raise SchemaError("STATE_STALE", "held receipt timestamp/freshness mismatch")
    return obj["receipt_sha256"]


_RUNTIME_RECEIPT_KEYS = {
    "schema_id",
    "runtime_state_id",
    "profile_receipt_sha256",
    "anchor_timestamp_ns",
    "freshness_deadline_ns",
    "role_instances_sha256",
    "resource_facts_sha256",
    "monitor_evidence_sha256",
    "admission_status",
    "receipt_sha256",
}


def validate_runtime_resource_receipt(
    receipt: object,
    *,
    expected_profile_receipt_sha256: str,
    anchor_timestamp_ns: int,
    now_ns: int,
) -> str:
    _strict_int(
        anchor_timestamp_ns,
        "RUNTIME_RECEIPT",
        "expected anchor_timestamp_ns",
        minimum=-(1 << 63),
        maximum=(1 << 63) - 1,
    )
    _strict_int(now_ns, "STATE_STALE", "now_ns", minimum=-(1 << 63), maximum=(1 << 63) - 1)
    obj = _validate_self_hashing_receipt(
        receipt,
        keys=_RUNTIME_RECEIPT_KEYS,
        schema_id="runtime_resource_receipt_v1",
        code="RUNTIME_RECEIPT",
    )
    _identifier(obj["runtime_state_id"], name="runtime_state_id", code="RUNTIME_RECEIPT")
    _sha_field(expected_profile_receipt_sha256, name="expected profile receipt", code="RUNTIME_RECEIPT")
    if obj["profile_receipt_sha256"] != expected_profile_receipt_sha256:
        raise SchemaError("RECEIPT_LAYER_VIOLATION", "runtime receipt attempts to replace static profile")
    for field in (
        "role_instances_sha256",
        "resource_facts_sha256",
        "monitor_evidence_sha256",
    ):
        _sha_field(obj[field], name=field, code="RUNTIME_RECEIPT")
    if obj["admission_status"] not in {"ADMITTED", "REJECTED"} or type(obj["admission_status"]) is not str:
        raise SchemaError("RUNTIME_RECEIPT", "admission status is not frozen enum")
    anchor = _strict_int(
        obj["anchor_timestamp_ns"], "RUNTIME_RECEIPT", "anchor_timestamp_ns", minimum=-(1 << 63), maximum=(1 << 63) - 1
    )
    deadline = _strict_int(
        obj["freshness_deadline_ns"], "RUNTIME_RECEIPT", "freshness_deadline_ns", minimum=-(1 << 63), maximum=(1 << 63) - 1
    )
    if anchor != anchor_timestamp_ns or deadline < anchor or now_ns > deadline:
        raise SchemaError("STATE_STALE", "runtime receipt timestamp/freshness mismatch")
    return obj["receipt_sha256"]


def _assert_frozen_constants() -> None:
    expected = {
        "TRANSLATIONS": 131_072,
        "L_NONVERIFY": 70_934_200_320,
        "L_VERIFY": 2_214_723_584,
        "N": 43_695_467_397_120,
        "V": 194_895_675_392,
        "B_NONVERIFY": 14_319_704_032_895_892_051_583_303_680,
        "B_VERIFY": 151_937_297_146_405_128_765_440,
        "STRUCTURAL_STATES": 157_516_896_299_151_958_972_545_105_921,
    }
    actual = {name: globals()[name] for name in expected}
    if actual != expected:
        raise RuntimeError(f"minimal-r1d frozen capacity mismatch: {actual!r}")
    if MAX_GLOBAL_RANK.bit_length() != MINIMUM_INDEX_BITS:
        raise RuntimeError("minimal-r1d index width mismatch")
    if math.gcd(EQUAL_BIT_MULTIPLIER, MAX_GLOBAL_RANK) != 1:
        raise RuntimeError("minimal-r1d equal-bit multiplier is not invertible")
    if (EQUAL_BIT_MULTIPLIER * EQUAL_BIT_INVERSE) % MAX_GLOBAL_RANK != 1:
        raise RuntimeError("minimal-r1d equal-bit inverse mismatch")
    if repr(math.log2(STRUCTURAL_STATES)) != STRUCTURAL_LOG2:
        raise RuntimeError("minimal-r1d structural log2 mismatch")


_assert_frozen_constants()


__all__ = [
    "AXIS_TABLE_UNIT_TOLERANCE",
    "AXIS_VERTICAL_EPSILON_SQUARED",
    "AXIS_CENTERS_LE_F64_SHA256",
    "Assignment",
    "B_NONVERIFY",
    "B_VERIFY",
    "ClaimMode",
    "Constraint",
    "ConstraintFunction",
    "ConstraintType",
    "Effect",
    "EventPacket",
    "FRAME_ORTHONORMAL_TOLERANCE",
    "FRAME_SURFACE_LE_F64_SHA256",
    "FrameBasis",
    "GoalProjection",
    "HeldPublicState",
    "Locator",
    "LocatorEvidence",
    "LocatorKind",
    "MAX_GLOBAL_RANK",
    "N",
    "Part",
    "PathPose",
    "PathPrimitive",
    "PATH_SURFACE_LE_F64_SHA256",
    "Phase",
    "PublicExecutionState",
    "PublicStateValidationPolicy",
    "RASTER_VERTICES",
    "Relation",
    "ResourceClaim",
    "ResourceValidationContext",
    "RigidTransform",
    "Role",
    "STRUCTURAL_STATES",
    "SchemaError",
    "TargetNamespace",
    "TargetSelector",
    "TopologyMember",
    "V",
    "actuator_ownership_mask",
    "canonical_json_bytes",
    "capacity_metadata",
    "combination_rank",
    "compact_to_field_wire",
    "decode_axis_center",
    "decode_frame",
    "decode_translation_center",
    "equal_bit_inverse",
    "equal_bit_permute",
    "field_wire_to_compact",
    "generated_axis_centers",
    "locator_rank",
    "pack_compact",
    "pack_wire",
    "parse_strict_json_bytes",
    "payload_rank",
    "profile_bundle_capacity",
    "project_transform",
    "quantize_axis",
    "quantize_depth",
    "quantize_extent",
    "quantize_metric_cell",
    "quantize_normalized_uv",
    "quantize_pixel_center",
    "quantize_roll",
    "rank_packet",
    "rank_topology",
    "rank_wire",
    "require_defined_codebook",
    "sample_path_pose",
    "shared_semantic_key_hash",
    "shared_semantic_key_hash_json",
    "text_goal_hash",
    "text_goal_hash_json",
    "topology_capacity",
    "unpack_compact",
    "unpack_wire",
    "unrank_combination",
    "unrank_locator",
    "unrank_packet",
    "unrank_payload",
    "unrank_topology",
    "unrank_wire",
    "validate_frame_basis",
    "validate_held_public_state_receipt",
    "validate_locator_evidence",
    "validate_metric_codebook",
    "validate_metric_codebook_receipt",
    "validate_metric_manifest",
    "validate_packet",
    "validate_physical_resource_claims",
    "validate_public_execution_state",
    "validate_public_state_validation_policy",
    "validate_resource_claim",
    "validate_resource_compatibility",
    "validate_resource_validation_context",
    "validate_runtime_resource_receipt",
    "validate_shared_semantic_key",
    "validate_shared_semantic_key_json",
    "validate_static_capability_profile_receipt",
    "validate_text_goal",
    "validate_text_goal_json",
    "validate_verify_hold_existing",
    "verify_frozen_geometry_golden_surfaces",
    "zero_roll_basis",
]
