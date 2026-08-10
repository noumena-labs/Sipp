import SippCoreBindings

/// Unregistered endpoint configuration accepted by `SippClient.add`.
public struct Endpoint: Sendable {
    let model: ManagedModel

    private init(model: ManagedModel) {
        self.model = model
    }

    /// Create a local endpoint from a managed model.
    public static func local(_ model: ManagedModel) -> Endpoint {
        Endpoint(model: model)
    }
}

/// Registered client-scoped endpoint identity.
public struct EndpointRef: Equatable, Hashable, Sendable {
    public let id: String

    init(id: String) {
        self.id = id
    }

    init(_ endpoint: FfiEndpointRef) {
        id = endpoint.id
    }
}

extension Endpoint {
    var bridgeValue: FfiEndpoint {
        FfiEndpoint(model: model.bridgeValue)
    }
}

extension ManagedModel {
    var bridgeValue: FfiManagedModel {
        FfiManagedModel(
            id: id,
            name: name,
            bytes: bytes,
            modality: modality.bridgeValue,
            status: status.bridgeValue
        )
    }
}

extension ModelModality {
    var bridgeValue: FfiModelModality {
        switch self {
        case .text:
            .text
        case .vision:
            .vision
        case .audio:
            .audio
        case .multimodal:
            .multimodal
        }
    }
}

extension ModelStatus {
    var bridgeValue: FfiModelStatus {
        switch self {
        case .ready:
            .ready
        case .needsProjector:
            .needsProjector
        case .broken:
            .broken
        }
    }
}
