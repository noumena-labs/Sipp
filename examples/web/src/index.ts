import { renderIndex, type ExampleLink } from './common.js';

const links: readonly ExampleLink[] = [
  { href: '/query.html', label: 'Query' },
  { href: '/chat.html', label: 'Chat' },
  { href: '/embed.html', label: 'Embed' },
  { href: '/remote_gateway_query.html', label: 'Gateway query' },
  { href: '/remote_gateway_chat.html', label: 'Gateway chat' },
  { href: '/remote_gateway_embed.html', label: 'Gateway embed' },
];

renderIndex('CogentLM Web Examples', links);
