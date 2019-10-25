import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const set = key => (state,val) =>{
  Vue.set(state,key,val);
};

export default new Vuex.Store({
  state: {
    settings:{
      teamNumber:1577,
      connectionType:0,
      ip:"",
      gateway:"",
      netmask:"",
      hostname: "Chameleon-vision"
    },
    pipeline:{
      exposure:0,
      brightness:0,
      orientation:0,
      hue:[0,15],
      saturation:[0,15],
      value:[0,25],
      erode:false,
      dilate:false,
      area:[0,12],
      ratio:[0,12],
      extent:[0,12],
      speckle:5,
      targetGrouping:0,
      targetIntersection:0,
      sortMode:0,
      isBinary:0
    },
    cameraSettings:{},
    resolutionList:[],
    port:1181,
    currentCameraIndex:0,
    currentPipelineIndex:0,
    cameraList:[],
    pipelineList:[],
    point:{}
  },
  mutations: {
    settings: set('settings'),
    pipeline: set('pipeline'),
    cameraSettings: set('cameraSettings'),
    resolutionList: set('resolutionList'),
    port: set('port'),
    currentCameraIndex: set('currentCameraIndex'),
    currentPipelineIndex: set('currentPipelineIndex'),
    cameraList: set('cameraList'),
    pipelineList: set('pipelineList'),
    point:set('point')
  },
  actions: {
    settings: state => state.settings,
    pipeline: state => state.pipeline,
    cameraSettings: state =>state.cameraSettings,
    resolutionList: state =>state.resolutionList,
    port: state =>state.port,
    currentCameraIndex: state =>state.currentCameraIndex,
    currentPipelineIndex: state =>state.currentPipelineIndex,
    cameraList: state =>state.cameraList,
    pipelineList: state =>state.pipelineList,
    point: state =>state.point,
    setPipeValues(state,obj){
      for(let i in obj){
        Vue.set(state.pipeline,i,obj[i]);
      }
    }
  }
})
