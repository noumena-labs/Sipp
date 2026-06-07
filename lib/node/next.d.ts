import type { EndpointDescriptor } from './index.js'

export type NextGatewayAuthorization =
  | 'none'
  | ((request: Request) => boolean | { authorized: true } | Promise<boolean | { authorized: true }>)

export interface NextGatewayOptions {
  readonly aliases: Readonly<Record<string, EndpointDescriptor>>
  readonly auth: NextGatewayAuthorization
  readonly maxRequestBytes?: number
}

export interface NextGatewayRouteContext {
  readonly params?:
    | { readonly operation?: string }
    | Promise<{ readonly operation?: string }>
}

export type NextGatewayHandler = (
  request: Request,
  context?: NextGatewayRouteContext,
) => Promise<Response>

/**
 * Create a Next.js App Router gateway handler for the Node.js runtime.
 *
 * Set `serverExternalPackages: ['@noumena-labs/cogentlm-server']` in
 * `next.config.js`.
 */
export declare function createNextGateway(options: NextGatewayOptions): NextGatewayHandler
