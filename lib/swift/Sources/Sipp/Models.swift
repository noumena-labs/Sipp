import SippCoreBindings

/// A model registered in the client's model store.
public struct ManagedModel: Equatable, Sendable {
    public let id: String
    public let name: String
    public let bytes: UInt64
    public let modality: ModelModality
    public let status: ModelStatus
}

/// The inference inputs supported by a managed model.
public enum ModelModality: Equatable, Sendable {
    case text
    case vision
}

/// The installation state of a managed model.
public enum ModelStatus: Equatable, Sendable {
    case ready
    case needsProjector
    case broken
}

extension ManagedModel {
    init(_ model: FfiManagedModel) {
        id = model.id
        name = model.name
        bytes = model.bytes
        modality = ModelModality(model.modality)
        status = ModelStatus(model.status)
    }
}

extension ModelModality {
    init(_ modality: FfiModelModality) {
        switch modality {
        case .text:
            self = .text
        case .vision:
            self = .vision
        }
    }
}

extension ModelStatus {
    init(_ status: FfiModelStatus) {
        switch status {
        case .ready:
            self = .ready
        case .needsProjector:
            self = .needsProjector
        case .broken:
            self = .broken
        }
    }
}
