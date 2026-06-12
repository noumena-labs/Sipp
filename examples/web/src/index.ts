import { renderIndex, type ExampleLink } from './common.js';

const links: readonly ExampleLink[] = [
  { href: '/query.html', label: 'Query' },
  { href: '/chat.html', label: 'Chat' },
  { href: '/embed.html', label: 'Embed' },
  { href: '/gateway_local.html', label: 'Gateway local' },
  { href: '/gateway_query.html', label: 'Gateway query' },
  { href: '/gateway_chat.html', label: 'Gateway chat' },
  { href: '/gateway_embed.html', label: 'Gateway embed' },
];

renderIndex('Sipp Web Examples', links);
