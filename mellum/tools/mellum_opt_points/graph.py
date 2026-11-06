"""Bounded per-operation graph cache; captured operators retain eager shapes."""
import torch

class RegionGraphs:
    def __init__(self,max_shapes=2,bucket=False):
        self.graphs={};self.shapes={};self.max_shapes=max_shapes;self.bucket=bucket
        self.counts=dict(captures=0,replays=0,fallbacks=0,static_input_bytes=0)
    def run(self,op,fn,*inputs):
        n=inputs[0].size(0)
        cap=1<<(n-1).bit_length() if self.bucket else n
        shapes=[(cap,*x.shape[1:]) for x in inputs]
        def trim(out):
            if cap==n:return out
            return tuple(x[:n] for x in out) if isinstance(out,tuple) else out[:n]
        key=(op,tuple((shape,tuple(x.stride()),x.dtype) for shape,x in zip(shapes,inputs)))
        if key in self.graphs:
            graph,static,out=self.graphs[key]
            for dst,src in zip(static,inputs):dst[:n].copy_(src)
            graph.replay();self.counts['replays']+=1
            return trim(out)
        if self.shapes.get(op,0)>=self.max_shapes:
            self.counts['fallbacks']+=1
            return fn(*inputs)
        # This cache is scoped to one model and captures only ordinary decode.
        # No tokenizer, routing decisions, prompt IDs or generated outputs are cached.
        static=tuple(torch.empty_strided(shape,x.stride(),device=x.device,dtype=x.dtype).zero_() for shape,x in zip(shapes,inputs))
        for dst,src in zip(static,inputs):dst[:n].copy_(src)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):warm=fn(*static)
        torch.cuda.current_stream().wait_stream(stream)
        del warm
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=fn(*static)
        self.graphs[key]=(graph,static,out);self.shapes[op]=self.shapes.get(op,0)+1
        self.counts['captures']+=1
        self.counts['static_input_bytes']+=sum(x.numel()*x.element_size() for x in static)
        graph.replay()
        return trim(out)
