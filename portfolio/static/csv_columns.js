/* Header suggestions only; server-side CSV parsing remains authoritative. */
(() => {
  function readCSVHeaders(source) {
    if(typeof source!=='string'||!source.trim())return [];
    source=source.replace(/^\uFEFF+/,'');
    const headers=[];let field='',quoted=false,closed=false;
    for(let i=0;i<source.length;i++){
      if(i>=65536)return [];
      const char=source[i];
      if(quoted){
        if(char==='"'){if(source[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}}
        else field+=char;
      }else if(char===','){
        headers.push(field.trim());field='';closed=false;if(headers.length>=50)return [];
      }else if(char==='\n'||char==='\r'){
        break;
      }else if(closed){return [];}
      else if(char==='"'&&!field){quoted=true;}
      else field+=char;
    }
    if(quoted)return [];
    headers.push(field.trim());
    if(headers.some(header=>!header)||new Set(headers).size!==headers.length)return [];
    return headers;
  }
  if(typeof module!=='undefined'&&module.exports){module.exports={readCSVHeaders};return;}
  document.addEventListener('DOMContentLoaded',()=>{
    const source=document.querySelector('#source');if(!source)return;
    const inputs=[...document.querySelectorAll('#date-column,#category-column,#amount-column,#target')].filter(input=>input.tagName==='INPUT'&&input.type==='text');
    if(!inputs.length)return;
    const list=document.createElement('datalist');list.id='csv-column-suggestions';document.querySelector('main').append(list);
    inputs.forEach(input=>{input.setAttribute('list',list.id);input.title='Type or choose a column name from your CSV';});
    let previous=null;
    const sync=()=>{
      if(!['profile','clean','sales','train','evaluate'].includes(document.body.dataset.app))return;
      const headers=readCSVHeaders(source.value),signature=JSON.stringify(headers);if(signature===previous)return;previous=signature;
      list.replaceChildren(...headers.map(header=>{const option=document.createElement('option');option.value=header;return option;}));
    };
    source.addEventListener('input',sync);
    document.addEventListener('source-loaded',sync);
    document.addEventListener('click',event=>{if(event.target.closest('#sample,[data-load-example]'))queueMicrotask(sync);});
    window.addEventListener('hashchange',()=>queueMicrotask(sync));
    sync();
  });
})();
