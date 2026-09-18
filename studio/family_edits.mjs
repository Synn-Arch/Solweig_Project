// SPDX-License-Identifier: GPL-3.0-only
//
// Family edit composition: convert a capability-driven editor draft into
// contract edit items for the transport that can carry them.
//
// WIRING ONLY — this module contains no physics vocabulary. The editable
// property names, bounds, and operations all come from the capability
// document; the only table here says which transport each adapter's edits
// ride:
//
//   tree-edits-v1        POST /api/v1/scenarios/{id}/edits        (trees)
//   universal-edits-v1   POST /api/v1/scenarios/{id}/edits/universal
//                         (meteorological_forcing, landcover_surface,
//                          building_geometry, model_receptor_parameters)
//
// Adapters with no transport entry (a document that marks an adapter
// integrated before this build wires it) are still fully editable as
// drafts: `composeFamilyEdit` throws a FamilyTransportError carrying the
// exact universal item the editor composed, so nothing is silently dropped
// and the payload is one transport-table row away from sendable.

import { adapterPropertyDefault, propertyControls, validateDraft } from "./capabilities.mjs";

export const TREE_TRANSPORT = "tree-edits-v1";
export const UNIVERSAL_TRANSPORT = "universal-edits-v1";

export const EDIT_TRANSPORTS = Object.freeze({
  vegetation_geometry: TREE_TRANSPORT,
  meteorological_forcing: UNIVERSAL_TRANSPORT,
  landcover_surface: UNIVERSAL_TRANSPORT,
  building_geometry: UNIVERSAL_TRANSPORT,
  model_receptor_parameters: UNIVERSAL_TRANSPORT,
});

/** The transport id for one adapter, or null when this build has none. */
export function editTransport(adapterId) {
  return Object.prototype.hasOwnProperty.call(EDIT_TRANSPORTS, adapterId)
    ? EDIT_TRANSPORTS[adapterId]
    : null;
}

/**
 * Where the document declares the inert default for a contract passthrough
 * the EDITED adapter's own schema rejects (wiring names, not physics values):
 * the vegetation schema fences `transmissivity` as a property and its
 * rejection reason points at `model_receptor_parameters.transVeg`, so the
 * passthrough value is READ from that sibling schema's declared default —
 * never restated as a local literal that could drift from the document.
 */
const PASS_THROUGH_DEFAULT_SOURCES = Object.freeze({
  vegetation_geometry: Object.freeze({
    adapter: "model_receptor_parameters",
    property: "transVeg",
  }),
});

/**
 * Thrown when an adapter's editor composed a valid draft but this build has
 * no transport entry for its family (a document that marks an adapter
 * integrated ahead of this frontend's wiring). `.payload` is the exact
 * universal edit item the frontend is ready to POST.
 */
export class FamilyTransportError extends Error {
  constructor(adapterId, payload) {
    super(
      `the edit endpoint of this server build has no transport for ` +
        `${adapterId} edits yet; composed payload is attached`,
    );
    this.name = "FamilyTransportError";
    this.adapterId = adapterId;
    this.payload = payload;
  }
}

export class DraftValidationError extends Error {
  constructor(errors) {
    super(
      `draft failed capability validation: ${errors
        .map((error) => `${error.property} (${error.message})`)
        .join("; ")}`,
    );
    this.name = "DraftValidationError";
    this.errors = errors;
  }
}

function roundTo(value, digits = 6) {
  return Number(Number(value).toFixed(digits));
}

/**
 * Compose the contract edit item for one vegetation draft.
 *
 * The capability schema speaks `canopy_radius_m`; the tree contract speaks
 * `canopy_diameter_m` — the ×2 conversion is the only unit mapping here.
 * Position stays in the contract's site-normalized u/v (canvas-driven).
 * `properties` arrives schema-validated (validateDraft has already filled
 * the schema defaults), so no local default restatements exist below.
 *
 * @param {object} draft {operation, treeId, u, v, properties:{height_m,
 *                 canopy_radius_m, trunk_ratio}, transmissivity?}
 * @param {number|null} transmissivityDefault document-declared inert default
 *                 for the passthrough the vegetation schema rejects
 */
function vegetationEditItem(draft, transmissivityDefault = null) {
  const properties = draft.properties ?? {};
  const tree = {
    tree_id: String(draft.treeId),
    component_type: "broad_canopy",
    u: roundTo(draft.u),
    v: roundTo(draft.v),
    height_m: roundTo(properties.height_m),
    canopy_diameter_m: roundTo(properties.canopy_radius_m * 2),
    trunk_ratio: roundTo(properties.trunk_ratio),
    phenology: "deciduous",
  };
  const transmissivity = draft.transmissivity ?? transmissivityDefault;
  // Omitted when neither the draft nor the document supplies a value: the
  // server's own model default then applies (schema-derived by construction).
  if (transmissivity !== null && transmissivity !== undefined) {
    tree.transmissivity = roundTo(transmissivity);
  }
  if (draft.label) tree.metadata = { label: draft.label };
  if (draft.operation === "delete") {
    return { operation: "delete", tree_id: String(draft.treeId) };
  }
  return { operation: draft.operation, tree };
}

/**
 * The universal edit item ({adapter, operation, values, target, time_index,
 * old_values}) for one validated draft — exactly the POST
 * /api/v1/scenarios/{id}/edits/universal contract grammar. All validity
 * rulings stay server-side (registry/adapter validators); this is field
 * placement only.
 */
function universalEditItem(adapterId, draft, values) {
  const item = {
    adapter: adapterId,
    operation: draft.operation ?? null,
    values,
    target: draft.target ?? null,
    time_index: null,
    old_values: null,
  };
  if (draft.timeIndex !== undefined && draft.timeIndex !== null) {
    item.time_index = Number(draft.timeIndex);
  }
  if (draft.oldValues !== undefined && draft.oldValues !== null) {
    item.old_values = draft.oldValues;
  }
  return item;
}

/**
 * Which schema-validated values a universal edit carries.
 *
 * `touchedProperties` (the editor's set of user-entered properties) keeps a
 * partial edit PARTIAL: untouched properties are omitted and stay at their
 * current scene values instead of being reverted to schema defaults. The
 * paint class is always carried — a class-picker family's whole payload is
 * the class. Drafts without touched-tracking send the full validated set
 * (meteorology's schema has no defaults, so its full row is required
 * either way).
 */
function universalValues(validation, draft) {
  const touched = Array.isArray(draft.touchedProperties)
    ? new Set(draft.touchedProperties.map(String))
    : null;
  if (touched === null) return validation.values;
  const values = {};
  for (const [name, value] of Object.entries(validation.values)) {
    if (name === "class" || touched.has(name)) values[name] = value;
  }
  return values;
}

/**
 * Compose edit items for one family draft, validating every value against
 * the adapter's property schema first.
 *
 * @param {object} capabilities parsed capability document
 * @param {string} adapterId adapter id from the document
 * @param {object} draft family-specific draft (see tests for the shapes);
 *        universal drafts may carry `touchedProperties` (string[]),
 *        `target` (building id / paint window), `timeIndex`, `oldValues`
 * @returns {Array} contract edit items for the adapter's transport
 * @throws {DraftValidationError} a value violates the document's bounds
 * @throws {FamilyTransportError} no transport for this family (yet)
 */
export function composeFamilyEdit(capabilities, adapterId, draft) {
  const adapter = capabilities.adapterById.get(adapterId);
  if (!adapter) {
    throw new DraftValidationError([
      { property: "adapter", message: `unknown adapter ${adapterId}` },
    ]);
  }
  const { controls, classPicker } = propertyControls(adapter, capabilities);
  // Deletes address the identity only; requiring property values for them
  // would make the contract's own delete item unsendable.
  const skipValues = draft.operation === "delete";
  const validation = skipValues
    ? { values: {}, errors: [] }
    : validateDraft(controls, draft.properties ?? {}, { classPicker });
  if (validation.errors.length > 0) {
    throw new DraftValidationError(validation.errors);
  }
  const transport = EDIT_TRANSPORTS[adapterId];
  if (transport === TREE_TRANSPORT) {
    const source = PASS_THROUGH_DEFAULT_SOURCES[adapterId];
    const transmissivityDefault = source
      ? adapterPropertyDefault(capabilities, source.adapter, source.property)
      : null;
    return [
      vegetationEditItem({ ...draft, properties: validation.values }, transmissivityDefault),
    ];
  }
  if (transport === UNIVERSAL_TRANSPORT) {
    const values = universalValues(validation, draft);
    // A delete addresses the identity only, so an empty value set is legal
    // exactly for it; every other operation must carry at least one value.
    if (!skipValues && Object.keys(values).length === 0) {
      throw new DraftValidationError([
        {
          property: "values",
          message:
            "set at least one property (untouched properties keep their current scene values)",
        },
      ]);
    }
    return [universalEditItem(adapterId, draft, values)];
  }
  // No transport entry: disclose the exact item this editor is ready to send
  // (values = the schema-validated draft, exactly the schema's vocabulary —
  // unknown draft properties were rejected above, never forwarded).
  throw new FamilyTransportError(
    adapterId,
    universalEditItem(adapterId, draft, validation.values),
  );
}
