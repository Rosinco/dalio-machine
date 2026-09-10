import { ReactFlow, Background, Controls, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { Pressure } from './types';

export default function PressureFlow({ pressure }: { pressure: Pressure }) {
  const nodes: Node[] = [{ id: 'constraint', position: { x: 10, y: 55 }, data: { label: pressure.constraint }, style: { width: 165, background: '#faf3d7', borderColor: '#c9b15b', fontSize: 11 } },
    ...pressure.forced_options.map((text, i) => ({ id: `option-${i}`, position: { x: 260, y: i * 85 }, data: { label: text }, style: { width: 160, fontSize: 11, borderColor: '#d9dfd4' } }))];
  const edges: Edge[] = pressure.forced_options.map((_, i) => ({ id: `edge-${i}`, source: 'constraint', target: `option-${i}`, style: { stroke: '#8396ad' } }));
  return <div className="flow" aria-label={`Rule diagram: ${pressure.title}`}><ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false} nodesConnectable={false} panOnScroll zoomOnScroll={false} minZoom={.3}><Background color="#cdd7ce" gap={18} /><Controls showInteractive={false} /></ReactFlow></div>;
}
